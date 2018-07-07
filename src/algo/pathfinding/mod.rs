use std::collections::HashMap;
use std::hash::Hash;

use algo::{Measure, FloatMeasure};
use visit::{IntoEdges, Visitable, NodeCount, IntoNodeIdentifiers, NodeIndexable};

mod astar;
mod bellman_ford;
mod dijkstra;
mod path;

pub use self::astar::Astar;
pub use self::bellman_ford::{BellmanFord, NegativeCycle};
pub use self::dijkstra::Dijkstra;

pub use self::path::{Path, IndexableNodeMap};

/// Builders used for type safe configuration of pathfinding algorithms. Mainly here to appear in
/// generated documentation.
pub mod builders {
    pub use super::astar::{AstarBuilder1, AstarBuilder2, AstarBuilder3, ConfiguredAstar};
    pub use super::bellman_ford::{BellmanFordBuilder1, BellmanFordBuilder2, ConfiguredBellmanFord};
    pub use super::dijkstra::{DijkstraBuilder1, DijkstraBuilder2, ConfiguredDijkstra};
}

/// Traits used to configure the pathfinding, such as predecessors or costs.
pub mod traits {
    pub use super::path::{CostMap, PredecessorMap, PredecessorMapConfigured};
}

/// [Generic] Dijkstra's shortest path algorithm.
///
/// Compute the length of the shortest path from `start` to every reachable
/// node.
///
/// The graph should be `Visitable` and implement `IntoEdges`. The function
/// `edge_cost` should return the cost for a particular edge, which is used
/// to compute path costs. Edge costs must be non-negative.
///
/// If `goal` is not `None`, then the algorithm terminates once the `goal` node's
/// cost is calculated.
///
/// Returns a `HashMap` that maps `NodeId` to path cost.
pub fn dijkstra<G, F, K>(graph: G,
                         start: G::NodeId,
                         goal: Option<G::NodeId>,
                         edge_cost: F)
                         -> HashMap<G::NodeId, K>
    where G: IntoEdges + Visitable,
          G::NodeId: Eq + Hash,
          F: Fn(G::EdgeRef) -> K,
          K: Measure + Copy
{
    let dijkstra = Dijkstra::new(graph)
        .edge_cost(edge_cost)
        .cost_map(HashMap::new());

    let path = {
        if let Some(goal) = goal {
            dijkstra.path(start, goal)
        } else {
            dijkstra.path_all(start)
        }
    };

    path.into_costs()
}

/// [Generic] A* shortest path algorithm.
///
/// Computes the shortest path from `start` to `finish`, including the total path cost.
///
/// `finish` is implicitly given via the `is_goal` callback, which should return `true` if the
/// given node is the finish node.
///
/// The function `edge_cost` should return the cost for a particular edge. Edge costs must be
/// non-negative.
///
/// The function `estimate_cost` should return the estimated cost to the finish for a particular
/// node. For the algorithm to find the actual shortest path, it should be admissible, meaning that
/// it should never overestimate the actual cost to get to the nearest goal node. Estimate costs
/// must also be non-negative.
///
/// The graph should be `Visitable` and implement `IntoEdges`.
///
/// ```
/// use petgraph::Graph;
/// use petgraph::algo::astar;
///
/// let mut g = Graph::new();
/// let a = g.add_node((0., 0.));
/// let b = g.add_node((2., 0.));
/// let c = g.add_node((1., 1.));
/// let d = g.add_node((0., 2.));
/// let e = g.add_node((3., 3.));
/// let f = g.add_node((4., 2.));
/// g.extend_with_edges(&[
///     (a, b, 2),
///     (a, d, 4),
///     (b, c, 1),
///     (b, f, 7),
///     (c, e, 5),
///     (e, f, 1),
///     (d, e, 1),
/// ]);
///
/// let path = astar(&g, a, |finish| finish == f, |e| *e.weight(), |_| 0);
/// assert_eq!(path, Some((6, vec![a, d, e, f])));
/// ```
///
/// Returns the total cost + the path of subsequent `NodeId` from start to finish, if one was
/// found.
pub fn astar<G, F, H, K, IsGoal>(graph: G,
                                 start: G::NodeId,
                                 is_goal: IsGoal,
                                 edge_cost: F,
                                 estimate_cost: H)
                                 -> Option<(K, Vec<G::NodeId>)>
    where G: IntoEdges + Visitable,
          IsGoal: Fn(G::NodeId) -> bool,
          G::NodeId: Eq + Hash,
          F: Fn(G::EdgeRef) -> K,
          H: Fn(G::NodeId) -> K,
          K: Measure + Copy
{
    Astar::new(graph)
        .edge_cost(edge_cost)
        .estimate_cost(estimate_cost)
        .cost_map(HashMap::new())
        .predecessor_map(HashMap::new())
        .path_with(start, is_goal)
        .into_nodes()
}

/// [Generic] Compute shortest paths from node `source` to all other.
///
/// Using the [Bellman–Ford algorithm][bf]; negative edge costs are
/// permitted, but the graph must not have a cycle of negative weights
/// (in that case it will return an error).
///
/// On success, return one vec with path costs, and another one which points
/// out the predecessor of a node along a shortest path. The vectors
/// are indexed by the graph's node indices.
///
/// [bf]: https://en.wikipedia.org/wiki/Bellman%E2%80%93Ford_algorithm
pub fn bellman_ford<G>(graph: G,
                       start: G::NodeId)
                       -> Result<(Vec<G::EdgeWeight>, Vec<Option<G::NodeId>>), NegativeCycle>
    where G: NodeCount + IntoNodeIdentifiers + IntoEdges + NodeIndexable,
          G::EdgeWeight: FloatMeasure
{
    let paths = BellmanFord::new(graph)
        .cost_map(IndexableNodeMap::new())
        .predecessor_map(IndexableNodeMap::new())
        .path_all(start);

    paths.map(|p| {
                  let (costs, predecessors, _) = p.unpack();
                  (costs.into_node_map(), predecessors.into_node_map())
              })
}

