"""DSGApi public interface.

This module defines the high-level Dynamic Scene Graph API (``DSGApi``) for
DSG-JIT. The intent is that this API can be used directly as a Python library
interface, or wrapped by a communication layer (HTTP/REST, gRPC, WebSocket,
ROS2, etc.) to expose DSG-JIT as a standalone service or edge node.

All methods are intentionally high-level, backend-agnostic, and
representation-agnostic. Implementations are expected to delegate to
``SceneGraphWorld``, a conforming FactorGraph backend, and optional storage
backends for multi-resolution geometric data (voxels, meshes, point clouds,
raw data, neural fields, etc.).

This file only defines method signatures and Sphinx-style docstrings; the
implementation details are left to concrete subclasses or future
implementations.
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union


# ---------------------------------------------------------------------------
# Type aliases (opaque to users of the API)
# ---------------------------------------------------------------------------

AgentId = str
NodeId = Any
ObjectId = Any
RoomId = Any
PlaceId = Any
RepId = str
TimeIndex = int
VersionId = Union[str, int]
Patch = Dict[str, Any]
SubgraphDict = Dict[str, Any]
ArrayLike = Any

#TODO update to utilize DSG not SceneGraphWorld

class DSGApi:
    """High-level Dynamic Scene Graph API.

    This class specifies the public DSG-JIT API that applications and
    services should depend on. It is designed to be:

    - High-level and semantic (agents, rooms, places, objects, etc.).
    - Backend-agnostic with respect to the FactorGraph implementation.
    - Representation-agnostic with respect to geometry (voxels, meshes,
      point clouds, raw data, neural fields).

    Concrete implementations are expected to:

    - Maintain an internal ``SceneGraphWorld`` or equivalent world model.
    - Use a conforming FactorGraph backend as defined in the
      ``FactorGraphStandard``.
    - Optionally interact with external storage for large geometric data.
    """

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """Initialize a DSGApi instance.

        Concrete implementations may accept configuration objects,
        world/scene-graph instances, FactorGraph backend instances,
        and storage registries via ``*args``/``**kwargs``.

        Parameters
        ----------
        *args : Any
            Positional arguments for backend-specific configuration.
        **kwargs : Any
            Keyword arguments for backend-specific configuration.
        """

    # ------------------------------------------------------------------
    # 2. Agent & Trajectory API
    # ------------------------------------------------------------------

    def register_agent(self, agent_id: AgentId, attrs: Optional[Dict[str, Any]] = None) -> AgentId:
        """Register an agent in the world.

        Implementations should create any necessary scene-graph entity and
        internal bookkeeping for the specified agent.

        Parameters
        ----------
        agent_id : AgentId
            Unique identifier for the agent (e.g. ``"robot0"``).
        attrs : dict, optional
            Optional metadata associated with the agent (e.g. type, color,
            capabilities).

        Returns
        -------
        AgentId
            The registered agent identifier (usually identical to the
            input ``agent_id``).
        """
        raise NotImplementedError

    def add_agent_pose(
        self,
        agent_id: AgentId,
        t: TimeIndex,
        pose_se3: ArrayLike,
        attrs: Optional[Dict[str, Any]] = None,
    ) -> NodeId:
        """Add an SE(3) pose node for an agent at a given time index.

        Parameters
        ----------
        agent_id : AgentId
            Identifier of the agent.
        t : TimeIndex
            Discrete time index (e.g. frame index).
        pose_se3 : ArrayLike
            6D se(3) pose vector or similar representation
            ``[tx, ty, tz, rx, ry, rz]``.
        attrs : dict, optional
            Optional metadata associated with this pose node.

        Returns
        -------
        NodeId
            Identifier of the created pose node.
        """
        raise NotImplementedError

    def add_agent_trajectory(
        self,
        agent_id: AgentId,
        poses_se3: Sequence[ArrayLike],
        start_t: TimeIndex = 0,
        add_odom: bool = True,
        default_dx: Optional[float] = None,
        weight: float = 1.0,
    ) -> List[NodeId]:
        """Add a contiguous SE(3) trajectory for an agent.

        Optionally connects consecutive poses with odometry factors.

        Parameters
        ----------
        agent_id : AgentId
            Identifier of the agent.
        poses_se3 : sequence of ArrayLike
            Sequence of se(3) pose vectors for the agent.
        start_t : TimeIndex, optional
            Starting time index for the first pose, by default ``0``.
        add_odom : bool, optional
            If ``True``, connect consecutive poses with SE(3) odometry
            factors, by default ``True``.
        default_dx : float, optional
            Optional default translation along x-axis used to define
            approximate odometry when explicit odom is not provided.
        weight : float, optional
            Optional weight for odometry factors, by default ``1.0``.

        Returns
        -------
        list of NodeId
            List of pose node identifiers corresponding to the trajectory.
        """
        raise NotImplementedError

    def get_agent_times(self, agent_id: AgentId) -> List[TimeIndex]:
        """Return sorted time indices for which the agent has poses.

        Parameters
        ----------
        agent_id : AgentId
            Identifier of the agent.

        Returns
        -------
        list of TimeIndex
            Sorted list of time indices where poses exist for this agent.
        """
        raise NotImplementedError

    def get_agent_pose_nodes(self, agent_id: AgentId) -> List[NodeId]:
        """Return pose node identifiers for an agent, ordered by time.

        Parameters
        ----------
        agent_id : AgentId
            Identifier of the agent.

        Returns
        -------
        list of NodeId
            List of pose node identifiers for the agent.
        """
        raise NotImplementedError

    def get_agent_trajectory(
        self,
        agent_id: AgentId,
        x_opt: Optional[ArrayLike] = None,
        index: Optional[Mapping[NodeId, Tuple[int, int]]] = None,
    ) -> ArrayLike:
        """Return an agent trajectory as a sequence of se(3) vectors.

        If an optimized state ``x_opt`` and index mapping ``index`` are
        provided, the values are read from the optimized state; otherwise
        the current variable values are used.

        Parameters
        ----------
        agent_id : AgentId
            Identifier of the agent.
        x_opt : ArrayLike, optional
            Optimized state vector, e.g. result of :meth:`optimize`.
        index : mapping, optional
            Mapping from ``NodeId`` to ``(start, dim)`` indices inside
            ``x_opt``.

        Returns
        -------
        ArrayLike
            Trajectory as an array of shape ``(T, 6)`` or equivalent.
        """
        raise NotImplementedError

    def get_all_trajectories(
        self,
        x_opt: Optional[ArrayLike] = None,
        index: Optional[Mapping[NodeId, Tuple[int, int]]] = None,
    ) -> Dict[AgentId, ArrayLike]:
        """Return trajectories for all registered agents.

        Parameters
        ----------
        x_opt : ArrayLike, optional
            Optimized state vector, e.g. result of :meth:`optimize`.
        index : mapping, optional
            Mapping from ``NodeId`` to ``(start, dim)`` indices inside
            ``x_opt``.

        Returns
        -------
        dict
            Mapping from ``AgentId`` to trajectories (arrays of se(3)
            values).
        """
        raise NotImplementedError

    # ------------------------------------------------------------------
    # 3. Semantic Scene Graph API
    # ------------------------------------------------------------------

    def add_room(self, name: str, attrs: Optional[Dict[str, Any]] = None) -> RoomId:
        """Create a room node.

        Parameters
        ----------
        name : str
            Human-readable name for the room.
        attrs : dict, optional
            Optional metadata (e.g. floor number, semantic label).

        Returns
        -------
        RoomId
            Identifier of the created room node.
        """
        raise NotImplementedError

    def set_room_bounds(self, room_id: RoomId, bounds: ArrayLike) -> None:
        """Attach geometric bounds to a room.

        Commonly, bounds will be an axis-aligned bounding box (AABB) in
        the form ``[xmin, ymin, zmin, xmax, ymax, zmax]``, but the
        representation is implementation-dependent.

        Parameters
        ----------
        room_id : RoomId
            Identifier of the room node.
        bounds : ArrayLike
            Geometric bounds of the room.
        """
        raise NotImplementedError

    def add_place(
        self,
        room_id: Optional[RoomId],
        position: ArrayLike,
        attrs: Optional[Dict[str, Any]] = None,
    ) -> PlaceId:
        """Create a place node at a given position.

        A place typically represents a local navigation waypoint or
        topological node in the environment.

        Parameters
        ----------
        room_id : RoomId, optional
            Room in which this place resides; if ``None``, the place is
            initially unattached to any room.
        position : ArrayLike
            Reference position of the place in world coordinates.
        attrs : dict, optional
            Optional metadata associated with the place.

        Returns
        -------
        PlaceId
            Identifier of the created place node.
        """
        raise NotImplementedError

    def attach_place_to_room(
        self,
        place_id: PlaceId,
        room_id: RoomId,
        rel_type: str = "contained_in",
    ) -> None:
        """Attach a place to a room with a semantic relation.

        Parameters
        ----------
        place_id : PlaceId
            Identifier of the place node.
        room_id : RoomId
            Identifier of the room node.
        rel_type : str, optional
            Relationship type, by default ``"contained_in"``.
        """
        raise NotImplementedError

    def add_object(
        self,
        name: str,
        place_id: Optional[PlaceId] = None,
        attrs: Optional[Dict[str, Any]] = None,
    ) -> ObjectId:
        """Create an object node.

        Parameters
        ----------
        name : str
            Human-readable name for the object.
        place_id : PlaceId, optional
            Place where this object resides; if ``None``, initially
            unattached to any place.
        attrs : dict, optional
            Optional metadata associated with the object (e.g. semantic
            class, size, material).

        Returns
        -------
        ObjectId
            Identifier of the created object node.
        """
        raise NotImplementedError

    def set_object_pose(self, object_id: ObjectId, pose_se3: ArrayLike) -> None:
        """Set or update the SE(3) pose of an object.

        Parameters
        ----------
        object_id : ObjectId
            Identifier of the object node.
        pose_se3 : ArrayLike
            SE(3) pose of the object in world coordinates.
        """
        raise NotImplementedError

    def attach_object_to_place(
        self,
        object_id: ObjectId,
        place_id: PlaceId,
        rel_type: str = "in",
    ) -> None:
        """Attach an object to a place with a semantic relation.

        Parameters
        ----------
        object_id : ObjectId
            Identifier of the object node.
        place_id : PlaceId
            Identifier of the place node.
        rel_type : str, optional
            Relationship type, by default ``"in"``.
        """
        raise NotImplementedError

    def set_support_relation(
        self,
        object_id: ObjectId,
        support_id: NodeId,
        rel_type: str = "supported_by",
    ) -> None:
        """Declare a support relationship between two nodes.

        Parameters
        ----------
        object_id : ObjectId
            Identifier of the supported object node.
        support_id : NodeId
            Identifier of the supporting node (e.g. a table).
        rel_type : str, optional
            Relationship type, by default ``"supported_by"``.
        """
        raise NotImplementedError

    def add_agent_node(self, agent_id: AgentId, attrs: Optional[Dict[str, Any]] = None) -> NodeId:
        """Create a semantic node representing an agent entity.

        This is distinct from the time-indexed pose nodes that represent
        the agent's trajectory.

        Parameters
        ----------
        agent_id : AgentId
            Identifier of the agent.
        attrs : dict, optional
            Optional metadata associated with the agent node.

        Returns
        -------
        NodeId
            Identifier of the created agent node.
        """
        raise NotImplementedError

    def add_relation(
        self,
        src_id: NodeId,
        dst_id: NodeId,
        rel_type: str,
        attrs: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Add a generic semantic relation between two nodes.

        Parameters
        ----------
        src_id : NodeId
            Source node identifier.
        dst_id : NodeId
            Destination node identifier.
        rel_type : str
            String describing the relation type.
        attrs : dict, optional
            Optional metadata associated with the relation.
        """
        raise NotImplementedError

    def get_relations(
        self,
        node_id: NodeId,
        rel_type: Optional[str] = None,
        direction: str = "both",
    ) -> List[Tuple[NodeId, str]]:
        """Return relations involving a given node.

        Parameters
        ----------
        node_id : NodeId
            Node identifier of interest.
        rel_type : str, optional
            If provided, filter by relation type.
        direction : str, optional
            One of ``"out"``, ``"in"``, or ``"both"`` indicating whether
            to return outgoing, incoming, or all relations, by default
            ``"both"``.

        Returns
        -------
        list of tuple
            List of ``(other_node_id, rel_type)`` pairs.
        """
        raise NotImplementedError

    def set_node_labels(self, node_id: NodeId, labels: Sequence[str]) -> None:
        """Assign one or more semantic labels to a node.

        Parameters
        ----------
        node_id : NodeId
            Identifier of the node.
        labels : sequence of str
            Semantic labels (classes, categories, tags) for the node.
        """
        raise NotImplementedError

    def get_node_labels(self, node_id: NodeId) -> List[str]:
        """Return semantic labels assigned to a node.

        Parameters
        ----------
        node_id : NodeId
            Identifier of the node.

        Returns
        -------
        list of str
            List of labels associated with the node.
        """
        raise NotImplementedError

    def find_nodes_by_label(
        self,
        label: str,
        node_type: Optional[str] = None,
    ) -> List[NodeId]:
        """Find nodes that have a given semantic label.

        Parameters
        ----------
        label : str
            Semantic label to search for.
        node_type : str, optional
            Optional node type filter (e.g. ``"object"``, ``"room"``,
            ``"place"``).

        Returns
        -------
        list of NodeId
            Node identifiers with the given label.
        """
        raise NotImplementedError

    def find_objects_by_class(self, class_label: str) -> List[ObjectId]:
        """Find all object nodes matching a given class label.

        Parameters
        ----------
        class_label : str
            Class label of interest (e.g. ``"chair"``, ``"mug"``).

        Returns
        -------
        list of ObjectId
            Object identifiers whose labels include ``class_label``.
        """
        raise NotImplementedError

    # ------------------------------------------------------------------
    # 4. Measurement & Factor API
    # ------------------------------------------------------------------

    def add_range_observation(
        self,
        agent_id: AgentId,
        t: TimeIndex,
        target_id: NodeId,
        measured_range: float,
        sigma: float = 0.1,
    ) -> None:
        """Add a range measurement from an agent pose to a target node.

        Parameters
        ----------
        agent_id : AgentId
            Identifier of the agent.
        t : TimeIndex
            Time index at which the range was measured.
        target_id : NodeId
            Identifier of the target node.
        measured_range : float
            Measured distance from the agent to the target.
        sigma : float, optional
            Standard deviation of the measurement noise, by default
            ``0.1``.
        """
        raise NotImplementedError

    def add_odom_se3(
        self,
        agent_id: AgentId,
        t0: TimeIndex,
        t1: TimeIndex,
        dx_se3: ArrayLike,
        weight: float = 1.0,
    ) -> None:
        """Connect two agent poses with a full SE(3) odometry factor.

        Parameters
        ----------
        agent_id : AgentId
            Identifier of the agent.
        t0 : TimeIndex
            Time index of the first pose.
        t1 : TimeIndex
            Time index of the second pose.
        dx_se3 : ArrayLike
            Relative SE(3) motion from pose at ``t0`` to pose at ``t1``.
        weight : float, optional
            Optional factor weight, by default ``1.0``.
        """
        raise NotImplementedError

    def add_odom_tx(
        self,
        agent_id: AgentId,
        t0: TimeIndex,
        t1: TimeIndex,
        dx: float,
        weight: float = 1.0,
    ) -> None:
        """Add a simple 1D odometry factor along the x-axis.

        Parameters
        ----------
        agent_id : AgentId
            Identifier of the agent.
        t0 : TimeIndex
            Time index of the first pose.
        t1 : TimeIndex
            Time index of the second pose.
        dx : float
            Expected translation along the x-axis.
        weight : float, optional
            Optional factor weight, by default ``1.0``.
        """
        raise NotImplementedError

    def add_voxel_observation(
        self,
        point_world: ArrayLike,
        value: float,
        sigma: float,
        attrs: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Add a voxel-style observation at a world point.

        This can be used for occupancy or signed distance observations
        depending on the chosen factor model.

        Parameters
        ----------
        point_world : ArrayLike
            3D point in world coordinates where the observation applies.
        value : float
            Observed value (e.g. occupancy probability or signed
            distance).
        sigma : float
            Standard deviation of the observation noise.
        attrs : dict, optional
            Optional metadata associated with the observation.
        """
        raise NotImplementedError

    def add_bearing_observation(self, *args: Any, **kwargs: Any) -> None:
        """Add a bearing (directional) observation.

        This is a placeholder for future visual or feature-based factors.
        Concrete implementations should define the exact signature and
        semantics.

        Parameters
        ----------
        *args : Any
            Backend-specific positional arguments.
        **kwargs : Any
            Backend-specific keyword arguments.
        """
        raise NotImplementedError

    def add_photometric_observation(self, *args: Any, **kwargs: Any) -> None:
        """Add a generic photometric observation factor.

        Typically used for direct visual SLAM or residuals over image
        intensities.

        Parameters
        ----------
        *args : Any
            Backend-specific positional arguments.
        **kwargs : Any
            Backend-specific keyword arguments.
        """
        raise NotImplementedError

    def add_nerf_photometric_observation(self, *args: Any, **kwargs: Any) -> None:
        """Add a NeRF-based photometric observation factor.

        This is a placeholder for future NeRF / neural field integration.

        Parameters
        ----------
        *args : Any
            Backend-specific positional arguments.
        **kwargs : Any
            Backend-specific keyword arguments.
        """
        raise NotImplementedError

    # ------------------------------------------------------------------
    # 5. Multi-Resolution Representation API
    # ------------------------------------------------------------------

    def attach_representation(
        self,
        node_id: NodeId,
        rep_type: str,
        data_ref: Any,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> RepId:
        """Attach a geometric or appearance representation to a node.

        Representations may include voxel grids, meshes, point clouds,
        raw data, or neural fields. The actual data is referenced by
        ``data_ref`` and typically stored in an external storage system
        or in-memory registry.

        Parameters
        ----------
        node_id : NodeId
            Node to which the representation should be attached.
        rep_type : str
            Representation type (e.g. ``"VOXEL_GRID"``, ``"MESH"``,
            ``"POINT_CLOUD"``, ``"RAW_DATA"``, ``"NEURAL_FIELD"``).
        data_ref : Any
            Backend-specific data reference (file path, URI, DB key,
            in-memory handle, etc.).
        metadata : dict, optional
            Optional metadata such as resolution, scale, frame, etc.

        Returns
        -------
        RepId
            Identifier (handle) for the attached representation.
        """
        raise NotImplementedError

    def list_representations(self, node_id: NodeId) -> List[RepId]:
        """List representation handles attached to a node.

        Parameters
        ----------
        node_id : NodeId
            Identifier of the node.

        Returns
        -------
        list of RepId
            Representation identifiers attached to the node.
        """
        raise NotImplementedError

    def get_representation(self, rep_id: RepId) -> Dict[str, Any]:
        """Return metadata for a representation handle.

        The heavy data (voxels, meshes, etc.) is not returned directly;
        instead, client code should use the stored ``data_ref`` to fetch
        or load it as needed.

        Parameters
        ----------
        rep_id : RepId
            Representation identifier.

        Returns
        -------
        dict
            Metadata dictionary for the representation (including at
            least ``rep_type`` and ``data_ref``).
        """
        raise NotImplementedError

    def remove_representation(self, rep_id: RepId) -> None:
        """Detach and remove a representation from the DSG.

        Parameters
        ----------
        rep_id : RepId
            Representation identifier to remove.
        """
        raise NotImplementedError

    # ------------------------------------------------------------------
    # 6. Query & Subgraph API
    # ------------------------------------------------------------------

    def get_subgraph(
        self,
        root_id: NodeId,
        depth: Optional[int] = None,
        layer_mask: Optional[Sequence[str]] = None,
    ) -> SubgraphDict:
        """Return a serializable subgraph view rooted at a node.

        Parameters
        ----------
        root_id : NodeId
            Root node identifier.
        depth : int, optional
            Maximum traversal depth from the root; if ``None``, no
            explicit depth limit is applied.
        layer_mask : sequence of str, optional
            Optional subset of semantic layers to include (e.g.
            ``["room", "place", "object"]``).

        Returns
        -------
        dict
            JSON-like dictionary representing the subgraph.
        """
        raise NotImplementedError

    def get_room_subgraph(self, room_id: RoomId) -> SubgraphDict:
        """Return a subgraph view for a room and its descendants.

        Parameters
        ----------
        room_id : RoomId
            Room node identifier.

        Returns
        -------
        dict
            JSON-like dictionary representing the room subgraph.
        """
        raise NotImplementedError

    def get_agent_local_subgraph(
        self,
        agent_id: AgentId,
        t: TimeIndex,
        radius: float,
        layers: Optional[Sequence[str]] = None,
    ) -> SubgraphDict:
        """Return a local subgraph around an agent pose.

        The subgraph includes nodes within a spatial radius of the
        agent's pose at time ``t``, optionally filtered by semantic
        layers.

        Parameters
        ----------
        agent_id : AgentId
            Identifier of the agent.
        t : TimeIndex
            Time index of the agent pose.
        radius : float
            Spatial radius around the agent's pose.
        layers : sequence of str, optional
            Optional subset of layers to include.

        Returns
        -------
        dict
            JSON-like dictionary representing the local subgraph.
        """
        raise NotImplementedError

    def get_fov_subgraph(
        self,
        agent_id: AgentId,
        t: TimeIndex,
        fov_params: Dict[str, Any],
        layers: Optional[Sequence[str]] = None,
    ) -> SubgraphDict:
        """Return a subgraph corresponding to an agent's field of view.

        The field of view (FOV) is specified via ``fov_params``, which
        typically describes a camera frustum (e.g. FOV angles, near/far
        planes, aspect ratio, intrinsics).

        Parameters
        ----------
        agent_id : AgentId
            Identifier of the agent.
        t : TimeIndex
            Time index of the agent pose.
        fov_params : dict
            Parameters describing the FOV frustum.
        layers : sequence of str, optional
            Optional subset of layers to include.

        Returns
        -------
        dict
            JSON-like dictionary representing the FOV subgraph.
        """
        raise NotImplementedError

    def get_camera_fov_subgraph(
        self,
        camera_pose_se3: ArrayLike,
        camera_model: Dict[str, Any],
        layers: Optional[Sequence[str]] = None,
    ) -> SubgraphDict:
        """Return a FOV-based subgraph for an arbitrary camera pose.

        Parameters
        ----------
        camera_pose_se3 : ArrayLike
            SE(3) pose of the camera in world coordinates.
        camera_model : dict
            Description of the camera model (e.g. intrinsics, FOV,
            near/far planes).
        layers : sequence of str, optional
            Optional subset of layers to include.

        Returns
        -------
        dict
            JSON-like dictionary representing the FOV subgraph.
        """
        raise NotImplementedError

    def find_objects_in_radius(
        self,
        center: ArrayLike,
        radius: float,
        class_filter: Optional[str] = None,
    ) -> List[ObjectId]:
        """Return objects within a spatial radius of a point.

        Parameters
        ----------
        center : ArrayLike
            Center point in world coordinates.
        radius : float
            Radius around the center point.
        class_filter : str, optional
            Optional semantic class label filter.

        Returns
        -------
        list of ObjectId
            Object identifiers within the specified radius.
        """
        raise NotImplementedError

    def find_places_in_radius(self, center: ArrayLike, radius: float) -> List[PlaceId]:
        """Return place nodes within a spatial radius of a point.

        Parameters
        ----------
        center : ArrayLike
            Center point in world coordinates.
        radius : float
            Radius around the center point.

        Returns
        -------
        list of PlaceId
            Place identifiers within the specified radius.
        """
        raise NotImplementedError

    def get_room_for_point(self, point: ArrayLike) -> Optional[RoomId]:
        """Return the room containing a given point, if any.

        Parameters
        ----------
        point : ArrayLike
            Point in world coordinates.

        Returns
        -------
        RoomId or None
            Identifier of the room containing the point, or ``None`` if
            none is found.
        """
        raise NotImplementedError

    def find_nearest_object_of_class(
        self,
        class_label: str,
        from_point: Optional[ArrayLike] = None,
        from_node_id: Optional[NodeId] = None,
        max_radius: Optional[float] = None,
    ) -> Optional[ObjectId]:
        """Find the nearest object of a given semantic class.

        Parameters
        ----------
        class_label : str
            Semantic class label of interest.
        from_point : ArrayLike, optional
            Optional world point to search from.
        from_node_id : NodeId, optional
            Optional node whose position serves as search origin.
        max_radius : float, optional
            Optional maximum search radius.

        Returns
        -------
        ObjectId or None
            Identifier of the nearest matching object, or ``None`` if
            none is found.
        """
        raise NotImplementedError

    def compute_node_descriptor(self, node_id: NodeId, method: str = "default") -> ArrayLike:
        """Compute a descriptor for a node.

        Descriptors can be used for loop closure, retrieval, or
        similarity queries. The underlying method is backend-specific.

        Parameters
        ----------
        node_id : NodeId
            Identifier of the node.
        method : str, optional
            Descriptor method name, by default ``"default"``.

        Returns
        -------
        ArrayLike
            Descriptor array for the node.
        """
        raise NotImplementedError

    def compute_subgraph_descriptor(
        self,
        root_id: NodeId,
        layers: Optional[Sequence[str]] = None,
        method: str = "default",
    ) -> ArrayLike:
        """Compute a descriptor over a subgraph.

        Parameters
        ----------
        root_id : NodeId
            Root node identifier for the subgraph.
        layers : sequence of str, optional
            Optional subset of layers to include.
        method : str, optional
            Descriptor method name, by default ``"default"``.

        Returns
        -------
        ArrayLike
            Descriptor array for the subgraph.
        """
        raise NotImplementedError

    def find_candidate_loop_closures(
        self,
        agent_id: AgentId,
        t: TimeIndex,
        threshold: float,
        max_results: int = 10,
    ) -> List[TimeIndex]:
        """Suggest potential loop closure candidates for an agent.

        Parameters
        ----------
        agent_id : AgentId
            Identifier of the agent.
        t : TimeIndex
            Time index at which to search for loop closures.
        threshold : float
            Similarity threshold for candidate selection.
        max_results : int, optional
            Maximum number of candidates to return, by default ``10``.

        Returns
        -------
        list of TimeIndex
            Time indices representing candidate loop closure locations.
        """
        raise NotImplementedError

    def list_visible_objects(
        self,
        agent_id: AgentId,
        t: TimeIndex,
        fov_params: Dict[str, Any],
        class_filter: Optional[str] = None,
    ) -> List[ObjectId]:
        """Return objects within an agent's field of view.

        Parameters
        ----------
        agent_id : AgentId
            Identifier of the agent.
        t : TimeIndex
            Time index of the agent pose.
        fov_params : dict
            Parameters describing the FOV frustum.
        class_filter : str, optional
            Optional semantic class label filter.

        Returns
        -------
        list of ObjectId
            Object identifiers that are visible in the field of view.
        """
        raise NotImplementedError

    def list_visible_nodes(
        self,
        agent_id: AgentId,
        t: TimeIndex,
        fov_params: Dict[str, Any],
        layer_mask: Optional[Sequence[str]] = None,
    ) -> List[NodeId]:
        """Return nodes within an agent's field of view.

        Parameters
        ----------
        agent_id : AgentId
            Identifier of the agent.
        t : TimeIndex
            Time index of the agent pose.
        fov_params : dict
            Parameters describing the FOV frustum.
        layer_mask : sequence of str, optional
            Optional subset of layers to include.

        Returns
        -------
        list of NodeId
            Node identifiers that are visible in the field of view.
        """
        raise NotImplementedError

    # ------------------------------------------------------------------
    # 7. Optimization & State API
    # ------------------------------------------------------------------

    def optimize(self, method: str = "gn_jit", max_iters: int = 10) -> Tuple[ArrayLike, Dict[NodeId, Tuple[int, int]]]:
        """Run a global optimization over the current factor graph.

        Parameters
        ----------
        method : str, optional
            Optimization method to use (e.g. ``"gn"``, ``"manifold_gn"``,
            ``"gn_jit"``), by default ``"gn_jit"``.
        max_iters : int, optional
            Maximum number of iterations, by default ``10``.

        Returns
        -------
        tuple
            ``(x_opt, index)`` where ``x_opt`` is the optimized state
            vector and ``index`` maps ``NodeId`` to ``(start, dim)``
            slices in ``x_opt``.
        """
        raise NotImplementedError

    def optimize_subgraph(
        self,
        subgraph: SubgraphDict,
        method: str = "gn",
        max_iters: int = 5,
    ) -> Tuple[ArrayLike, Dict[NodeId, Tuple[int, int]]]:
        """Run a local optimization over a specific subgraph.

        Parameters
        ----------
        subgraph : dict
            Subgraph view (as returned by :meth:`get_subgraph`).
        method : str, optional
            Optimization method, by default ``"gn"``.
        max_iters : int, optional
            Maximum number of iterations, by default ``5``.

        Returns
        -------
        tuple
            ``(x_opt_local, index_local)`` where ``x_opt_local`` is the
            optimized state for the subgraph and ``index_local`` maps
            ``NodeId`` to slices in ``x_opt_local``.
        """
        raise NotImplementedError

    def get_variable_value(
        self,
        node_id: NodeId,
        x_opt: Optional[ArrayLike] = None,
        index: Optional[Mapping[NodeId, Tuple[int, int]]] = None,
    ) -> ArrayLike:
        """Get the current or optimized numeric state of a variable.

        If ``x_opt`` and ``index`` are provided, the value is read from
        the optimized state; otherwise the current stored value is
        returned.

        Parameters
        ----------
        node_id : NodeId
            Identifier of the variable node.
        x_opt : ArrayLike, optional
            Optimized state vector.
        index : mapping, optional
            Mapping from ``NodeId`` to ``(start, dim)`` slices in
            ``x_opt``.

        Returns
        -------
        ArrayLike
            Numeric state vector of the variable.
        """
        raise NotImplementedError

    def set_variable_value(self, node_id: NodeId, value: ArrayLike) -> None:
        """Set the numeric state vector of a variable.

        Parameters
        ----------
        node_id : NodeId
            Identifier of the variable node.
        value : ArrayLike
            New state vector for the variable.
        """
        raise NotImplementedError

    # ------------------------------------------------------------------
    # 8. I/O, Serialization, and Versioning API
    # ------------------------------------------------------------------

    def to_dict(self, subgraph: Optional[SubgraphDict] = None) -> Dict[str, Any]:
        """Serialize the DSG (or a subgraph) to a dictionary.

        Parameters
        ----------
        subgraph : dict, optional
            Optional subgraph view; if provided, only this subgraph is
            serialized.

        Returns
        -------
        dict
            JSON-like dictionary representing the DSG or subgraph.
        """
        raise NotImplementedError

    def from_dict(self, data: Mapping[str, Any]) -> None:
        """Load DSG state from a dictionary snapshot.

        Parameters
        ----------
        data : mapping
            Dictionary snapshot previously produced by :meth:`to_dict`.
        """
        raise NotImplementedError

    def checkpoint(self, label: str) -> VersionId:
        """Save a named checkpoint of the current DSG state.

        Parameters
        ----------
        label : str
            Human-readable label for this checkpoint.

        Returns
        -------
        VersionId
            Identifier for the created checkpoint.
        """
        raise NotImplementedError

    def list_checkpoints(self) -> List[VersionId]:
        """List available checkpoints.

        Returns
        -------
        list of VersionId
            Identifiers for stored checkpoints.
        """
        raise NotImplementedError

    def rollback_to(self, version_id: VersionId) -> None:
        """Restore DSG state to a previous checkpoint.

        Parameters
        ----------
        version_id : VersionId
            Identifier of the checkpoint to restore.
        """
        raise NotImplementedError

    def compute_diff(self, old_version_id: VersionId, new_version_id: VersionId) -> Patch:
        """Compute a patch representing the difference between versions.

        Parameters
        ----------
        old_version_id : VersionId
            Identifier of the old version.
        new_version_id : VersionId
            Identifier of the new version.

        Returns
        -------
        Patch
            Patch object representing the difference between the two
            versions.
        """
        raise NotImplementedError

    def apply_patch(self, patch: Patch) -> None:
        """Apply a patch to the current DSG state.

        Parameters
        ----------
        patch : Patch
            Patch object previously created by :meth:`compute_diff`.
        """
        raise NotImplementedError

    # ------------------------------------------------------------------
    # 9. Multi-Agent & Session Management API (Optional)
    # ------------------------------------------------------------------

    def get_agents(self) -> List[AgentId]:
        """Return all registered agents.

        Returns
        -------
        list of AgentId
            Identifiers of registered agents.
        """
        raise NotImplementedError

    def tag_session(self, session_id: str) -> None:
        """Tag subsequent changes as belonging to a given session.

        Parameters
        ----------
        session_id : str
            Session identifier.
        """
        raise NotImplementedError

    def get_sessions(self) -> List[str]:
        """Return known session identifiers.

        Returns
        -------
        list of str
            List of session identifiers.
        """
        raise NotImplementedError

    def get_session_subgraph(self, session_id: str) -> SubgraphDict:
        """Return a subgraph corresponding to a particular session.

        Parameters
        ----------
        session_id : str
            Session identifier.

        Returns
        -------
        dict
            JSON-like dictionary representing the session subgraph.
        """
        raise NotImplementedError

    def merge_sessions(self, session_ids: Sequence[str], strategy: str = "union") -> None:
        """Merge multiple session graphs according to a strategy.

        Parameters
        ----------
        session_ids : sequence of str
            Session identifiers to merge.
        strategy : str, optional
            Merge strategy (e.g. ``"union"``), by default ``"union"``.
        """
        raise NotImplementedError

    # ------------------------------------------------------------------
    # 10. Planning & Navigation Support API (Optional)
    # ------------------------------------------------------------------

    def get_topological_neighbors(
        self,
        node_id: NodeId,
        layer_mask: Optional[Sequence[str]] = None,
        edge_type: Optional[str] = None,
    ) -> List[NodeId]:
        """Return neighboring nodes in the topological sense.

        Parameters
        ----------
        node_id : NodeId
            Node identifier of interest.
        layer_mask : sequence of str, optional
            Optional subset of layers to include.
        edge_type : str, optional
            Optional relation/edge type filter.

        Returns
        -------
        list of NodeId
            Neighboring node identifiers.
        """
        raise NotImplementedError

    def plan_topological_path(
        self,
        start_place_id: PlaceId,
        goal_place_id: PlaceId,
        cost_mode: str = "shortest",
    ) -> List[PlaceId]:
        """Compute a high-level path over the place/room graph.

        Parameters
        ----------
        start_place_id : PlaceId
            Starting place identifier.
        goal_place_id : PlaceId
            Goal place identifier.
        cost_mode : str, optional
            Cost mode (e.g. ``"shortest"``), by default ``"shortest"``.

        Returns
        -------
        list of PlaceId
            Sequence of place identifiers representing the topological
            path.
        """
        raise NotImplementedError

    def plan_semantic_path(
        self,
        agent_id: AgentId,
        start_time: TimeIndex,
        goal_query: str,
        constraints: Optional[Dict[str, Any]] = None,
    ) -> List[NodeId]:
        """Plan a high-level semantic path for an agent.

        This method provides a hook for semantic or language-conditioned
        planning. The DSGApi itself does not need to implement language
        models; instead, this method defines a stable integration point
        for external planners.

        Parameters
        ----------
        agent_id : AgentId
            Identifier of the agent.
        start_time : TimeIndex
            Time index at which the plan should start.
        goal_query : str
            Free-form description of the goal (e.g. ``"go to the nearest
            mug in the kitchen"``).
        constraints : dict, optional
            Optional planning constraints.

        Returns
        -------
        list of NodeId
            Sequence of node identifiers representing a semantic plan.
        """
        raise NotImplementedError
