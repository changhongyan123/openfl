# Copyright (C) 2020-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""openfl.experimental.utilities.metaflow_utils module."""

from datetime import datetime
from metaflow.metaflow_environment import MetaflowEnvironment
from metaflow.plugins import LocalMetadataProvider
from metaflow.datastore import FlowDataStore, DATASTORES
from metaflow.graph import DAGNode, FlowGraph, StepVisitor
from metaflow.graph import deindent_docstring
from metaflow.datastore.task_datastore import TaskDataStore
from metaflow.datastore.exceptions import DataException, UnpicklableArtifactException
from metaflow.datastore.task_datastore import only_if_not_done, require_mode
import cloudpickle as pickle
import multiprocessing
import ray
import ast
import inspect
from pathlib import Path
from metaflow.runtime import TruncatedBuffer, mflog_msg, MAX_LOG_SIZE
from metaflow.mflog import mflog, RUNTIME_LOG_SOURCE
from metaflow.task import MetaDatum

class Flow:
    def __init__(self, name):
        """Mock flow for metaflow internals"""
        self.name = name


@ray.remote
class Counter(object):
    def __init__(self):
        self.value = 0

    def increment(self):
        self.value += 1
        return self.value

    def get_counter(self):
        return self.value


class DAGnode(DAGNode):
    def __init__(self, func_ast, decos, doc):
        self.name = func_ast.name
        self.func_lineno = func_ast.lineno
        self.decorators = decos
        self.doc = deindent_docstring(doc)
        self.parallel_step = any(
            getattr(deco, "IS_PARALLEL", False) for deco in decos)

        # these attributes are populated by _parse
        self.tail_next_lineno = 0
        self.type = None
        self.out_funcs = []
        self.has_tail_next = False
        self.invalid_tail_next = False
        self.num_args = 0
        self.foreach_param = None
        self.num_parallel = 0
        self.parallel_foreach = False
        self._parse(func_ast)

        # these attributes are populated by _traverse_graph
        self.in_funcs = set()
        self.split_parents = []
        self.matching_join = None
        # these attributes are populated by _postprocess
        self.is_inside_foreach = False

    def _parse(self, func_ast):
        self.num_args = len(func_ast.args.args)
        tail = func_ast.body[-1]

        # end doesn't need a transition
        if self.name == "end":
            # TYPE: end
            self.type = "end"

        # ensure that the tail an expression
        if not isinstance(tail, ast.Expr):
            return

        # determine the type of self.next transition
        try:
            if not self._expr_str(tail.value.func) == "self.next":
                return

            self.has_tail_next = True
            self.invalid_tail_next = True
            self.tail_next_lineno = tail.lineno
            self.out_funcs = [e.attr for e in tail.value.args]

            keywords = dict(
                (k.arg, getattr(k.value, "s", None)) for k in tail.value.keywords
            )
            # Second condition in the folliwing line added,
            # To add the support for up to 2 keyword arguments in Flowgraph
            if len(keywords) == 1 or len(keywords) == 2:
                if "foreach" in keywords:
                    # TYPE: foreach
                    self.type = "foreach"
                    if len(self.out_funcs) == 1:
                        self.foreach_param = keywords["foreach"]
                        self.invalid_tail_next = False
                elif "num_parallel" in keywords:
                    self.type = "foreach"
                    self.parallel_foreach = True
                    if len(self.out_funcs) == 1:
                        self.num_parallel = keywords["num_parallel"]
                        self.invalid_tail_next = False
                # Following else parse is also added,
                # If there are no keyword arguments it is a linear flow
                else:
                    self.type = "linear"
            elif len(keywords) == 0:
                if len(self.out_funcs) > 1:
                    # TYPE: split
                    self.type = "split"
                    self.invalid_tail_next = False
                elif len(self.out_funcs) == 1:
                    # TYPE: linear
                    if self.name == "start":
                        self.type = "start"
                    elif self.num_args > 1:
                        self.type = "join"
                    else:
                        self.type = "linear"
                    self.invalid_tail_next = False
        except AttributeError:
            return


class StepVisitor(StepVisitor):
    def __init__(self, nodes, flow):
        super().__init__(nodes, flow)

    def visit_FunctionDef(self, node):
        func = getattr(self.flow, node.name)
        if hasattr(func, "is_step"):
            self.nodes[node.name] = DAGnode(
                node, func.decorators, func.__doc__)


class FlowGraph(FlowGraph):
    def __init__(self, flow):
        self.name = flow.__name__
        self.nodes = self._create_nodes(flow)
        self.doc = deindent_docstring(flow.__doc__)
        self._traverse_graph()
        self._postprocess()

    def _create_nodes(self, flow):
        module = __import__(flow.__module__)
        tree = ast.parse(inspect.getsource(module)).body
        root = [n for n in tree if isinstance(n, ast.ClassDef) and n.name == self.name][
            0
        ]
        nodes = {}
        StepVisitor(nodes, flow).visit(root)
        return nodes


class TaskDataStore(TaskDataStore):
    def __init__(self, flow_datastore, run_id, step_name, task_id, attempt=None,
                 data_metadata=None, mode="r", allow_not_done=False):
        super().__init__(flow_datastore, run_id, step_name,
                         task_id, attempt, data_metadata, mode, allow_not_done)

    @only_if_not_done
    @require_mode("w")
    def save_artifacts(self, artifacts_iter, force_v4=False, len_hint=0):
        """
        Saves Metaflow Artifacts (Python objects) to the datastore and stores
        any relevant metadata needed to retrieve them.

        Typically, objects are pickled but the datastore may perform any
        operation that it deems necessary. You should only access artifacts
        using load_artifacts

        This method requires mode 'w'.

        Parameters
        ----------
        artifacts : Iterator[(string, object)]
            Iterator over the human-readable name of the object to save
            and the object itself
        force_v4 : boolean or Dict[string -> boolean]
            Indicates whether the artifact should be pickled using the v4
            version of pickle. If a single boolean, applies to all artifacts.
            If a dictionary, applies to the object named only. Defaults to False
            if not present or not specified
        len_hint: integer
            Estimated number of items in artifacts_iter
        """
        artifact_names = []

        def pickle_iter():
            for name, obj in artifacts_iter:
                do_v4 = (
                    force_v4 and force_v4
                    if isinstance(force_v4, bool)
                    else force_v4.get(name, False)
                )
                if do_v4:
                    encode_type = "gzip+pickle-v4"
                    if encode_type not in self._encodings:
                        raise DataException(
                            "Artifact *%s* requires a serialization encoding that "
                            "requires Python 3.4 or newer." % name
                        )
                    try:
                        blob = pickle.dumps(obj, protocol=4)
                    except TypeError as e:
                        raise UnpicklableArtifactException(name)
                else:
                    try:
                        blob = pickle.dumps(obj, protocol=2)
                        encode_type = "gzip+pickle-v2"
                    except (SystemError, OverflowError):
                        encode_type = "gzip+pickle-v4"
                        if encode_type not in self._encodings:
                            raise DataException(
                                "Artifact *%s* is very large (over 2GB). "
                                "You need to use Python 3.4 or newer if you want to "
                                "serialize large objects." % name
                            )
                        try:
                            blob = pickle.dumps(obj, protocol=4)
                        except TypeError as e:
                            raise UnpicklableArtifactException(name)
                    except TypeError as e:
                        raise UnpicklableArtifactException(name)

                self._info[name] = {
                    "size": len(blob),
                    "type": str(type(obj)),
                    "encoding": encode_type,
                }
                artifact_names.append(name)
                yield blob

        # Use the content-addressed store to store all artifacts
        save_result = self._ca_store.save_blobs(
            pickle_iter(), len_hint=len_hint)
        for name, result in zip(artifact_names, save_result):
            self._objects[name] = result.key


class FlowDataStore(FlowDataStore):
    def __init__(self, flow_name, environment, metadata=None, event_logger=None,
                 monitor=None, storage_impl=None, ds_root=None):
        super().__init__(flow_name, environment, metadata,
                         event_logger, monitor, storage_impl, ds_root)

    def get_task_datastore(
        self,
        run_id,
        step_name,
        task_id,
        attempt=None,
        data_metadata=None,
        mode="r",
        allow_not_done=False,
    ):

        return TaskDataStore(
            self,
            run_id,
            step_name,
            task_id,
            attempt=attempt,
            data_metadata=data_metadata,
            mode=mode,
            allow_not_done=allow_not_done,
        )


class MetaflowInterface:
    def __init__(self, flow, backend='ray'):
        self.backend = backend
        self.flow_name = flow.__name__
        #self._graph = FlowGraph(flow)
        #self._steps = [getattr(flow, node.name) for node in self._graph]
        if backend == 'ray':
            self.counter = Counter.remote()
        else:
            self.counter = 0

    def create_run(self):
        flow = Flow(self.flow_name)
        env = MetaflowEnvironment(self.flow_name)
        env.get_environment_info()
        self.local_metadata = LocalMetadataProvider(env, flow, None, None)
        self.run_id = self.local_metadata.new_run_id()
        self.flow_datastore = FlowDataStore(
            self.flow_name, env, metadata=self.local_metadata, storage_impl=DATASTORES['local'],
            ds_root=f'{Path.home()}/.metaflow')
        return self.run_id

    def create_task(self, task_name):
        # May need a lock here
        if self.backend == 'ray':
            task_id = ray.get(self.counter.get_counter.remote())
        else:
            task_id = self.counter
        self.local_metadata._task_id_seq = task_id
        self.local_metadata.new_task_id(self.run_id, task_name)
        if self.backend == 'ray':
            return ray.get(self.counter.increment.remote())
        else:
            self.counter += 1
            return self.counter

    def save_artifacts(self, data_pairs, task_name, task_id, buffer_out, buffer_err):
        """Use metaflow task datastore to save federated flow attributes"""
        task_datastore = self.flow_datastore.get_task_datastore(
            self.run_id, task_name, str(task_id), attempt=0, mode='w')
        task_datastore.init_task()
        task_datastore.save_artifacts(data_pairs)

        # Register metadata for task
        retry_count = 0
        metadata_tags = ["attempt_id:{0}".format(retry_count)]
        self.local_metadata.register_metadata(
            self.run_id,
            task_name,
            str(task_id),
            [
                MetaDatum(
                    field="attempt",
                    value=str(retry_count),
                    type="attempt",
                    tags=metadata_tags,
                ),
                MetaDatum(
                    field="origin-run-id",
                    value=str(0),
                    type="origin-run-id",
                    tags=metadata_tags,
                ),
                MetaDatum(
                    field="ds-type",
                    value=self.flow_datastore.TYPE,
                    type="ds-type",
                    tags=metadata_tags,
                ),
                MetaDatum(
                    field="ds-root",
                    value=self.flow_datastore.datastore_root,
                    type="ds-root",
                    tags=metadata_tags,
                ),
            ],
        )

        self.emit_log(buffer_out, buffer_err, task_datastore)

        task_datastore.done()

    def load_artifacts(self, artifact_names, task_name, task_id):
        """Use metaflow task datastore to load flow attributes"""
        task_datastore = self.flow_datastore.get_task_datastore(
            self.run_id, task_name, str(task_id), attempt=0, mode='r')
        return task_datastore.load_artifacts(artifact_names)

    def emit_log(self, msgbuffer_out, msgbuffer_err, task_datastore, system_msg=False):
        """
        This function writes the stdout and stderr to Metaflow TaskDatastore
        Args
            msgbuffer_out: StringIO buffer containing stdout
            msgbuffer_err: StringIO buffer containing stderr
            task_datastore: Metaflow TaskDataStore instance
        """
        stdout_buffer = TruncatedBuffer("stdout", MAX_LOG_SIZE)
        stderr_buffer = TruncatedBuffer("stderr", MAX_LOG_SIZE)

        for std_output in msgbuffer_out.readlines():
            timestamp = datetime.utcnow()
            stdout_buffer.write(mflog_msg(std_output, now=timestamp), system_msg=system_msg)

        for std_error in msgbuffer_err.readlines():
            timestamp = datetime.utcnow()
            stderr_buffer.write(mflog_msg(std_error, now=timestamp), system_msg=system_msg)

        task_datastore.save_logs(RUNTIME_LOG_SOURCE, {
            "stdout": stdout_buffer.get_buffer(),
            "stderr": stderr_buffer.get_buffer(),
            })
    