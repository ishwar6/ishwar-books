---
title: "HNSW, From the Ground Up"
description: "How Hierarchical Navigable Small World graphs find nearest neighbours in microseconds: the two ideas they combine, how the graph is built, what M, efConstruction and efSearch really cost, and measured numbers from a 200,000-vector index."
date: 2026-09-27
tags: [vector-search, hnsw, retrieval]
---

Every vector database you are likely to use, from Qdrant and Weaviate to pgvector and Elasticsearch, answers nearest-neighbour queries with the same algorithm: **HNSW**, Hierarchical Navigable Small World graphs. It is the default for a reason. On the index measured in this piece it returns 93% of the true top 10 in **0.085 ms**, where checking every vector takes **1.39 ms**, and the gap widens with every vector you add.

It is also an algorithm most people use as a black box with three knobs. This piece opens the box. HNSW is two older ideas glued together, and once you see both, every parameter stops being magic and becomes a trade you can reason about.

## The problem: exact search does not scale

Given a query vector, find the $$k$$ stored vectors closest to it. The exact answer means computing the distance to every one of the $$N$$ stored vectors: $$O(N \cdot d)$$ work per query. At 200,000 vectors of 128 dimensions that is already 25.6 million multiply-adds per query; at 100 million vectors it is hopeless for anything interactive.

**Approximate** nearest-neighbour (ANN) search gives up a little accuracy for a lot of speed. Its quality is measured as **recall@k**: of the true $$k$$ nearest neighbours, what fraction did the index return? Recall 0.95 at $$k = 10$$ means that, on average, 9.5 of the 10 results are the true top 10.

ANN methods come in a few families:

| Family | Idea | Examples |
|---|---|---|
| Trees | recursively split the space | k-d trees, Annoy |
| Hashing | hash similar vectors into the same bucket | LSH |
| Quantisation | compress vectors and compare the codes | IVF, product quantisation |
| **Graphs** | connect each vector to its neighbours, then walk the graph | NSW, **HNSW**, Vamana |

Trees break down in high dimensions and hashing needs many tables to reach high recall. Graphs have dominated public ANN benchmarks for years, and HNSW is the graph method everyone ships.

## Idea 1: the skip list

A **skip list** (William Pugh, 1990) is a sorted linked list with express lanes. The bottom level holds every element in order. Each level above holds a random subset of the level below (typically half), so the top levels have few elements and links that jump far.

<figure class="fig"><svg viewBox="0 0 740 230" role="img" aria-label="A skip list with three levels; searching for 6 skips from 1 to 5 on the top level, drops down twice, then steps from 5 to 6 on the bottom level."><defs><marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path class="arrow" d="M0,0L10,5L0,10z"/></marker><marker id="ah-on" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path class="arrow-on" d="M0,0L10,5L0,10z"/></marker></defs><text class="t-label" x="8" y="45">LEVEL 2</text><text class="t-label" x="8" y="120">LEVEL 1</text><text class="t-label" x="8" y="195">LEVEL 0</text><line class="drop" x1="110" y1="56" x2="110" y2="174"/><line class="drop" x1="254" y1="131" x2="254" y2="174"/><line class="drop" x1="398" y1="56" x2="398" y2="174"/><line class="drop" x1="542" y1="131" x2="542" y2="174"/><line class="drop" x1="686" y1="56" x2="686" y2="174"/><line class="path" x1="130" y1="40" x2="376" y2="40" marker-end="url(#ah-on)"/><line class="edge" x1="418" y1="40" x2="664" y2="40" marker-end="url(#ah)"/><line class="edge" x1="130" y1="115" x2="232" y2="115" marker-end="url(#ah)"/><line class="edge" x1="274" y1="115" x2="376" y2="115" marker-end="url(#ah)"/><line class="edge" x1="418" y1="115" x2="520" y2="115" marker-end="url(#ah)"/><line class="edge" x1="562" y1="115" x2="664" y2="115" marker-end="url(#ah)"/><line class="edge" x1="130" y1="190" x2="160" y2="190" marker-end="url(#ah)"/><line class="edge" x1="202" y1="190" x2="232" y2="190" marker-end="url(#ah)"/><line class="edge" x1="274" y1="190" x2="304" y2="190" marker-end="url(#ah)"/><line class="edge" x1="346" y1="190" x2="376" y2="190" marker-end="url(#ah)"/><line class="path" x1="418" y1="190" x2="448" y2="190" marker-end="url(#ah-on)"/><line class="edge" x1="490" y1="190" x2="520" y2="190" marker-end="url(#ah)"/><line class="edge" x1="562" y1="190" x2="592" y2="190" marker-end="url(#ah)"/><line class="edge" x1="634" y1="190" x2="664" y2="190" marker-end="url(#ah)"/><line class="path" x1="398" y1="56" x2="398" y2="97" marker-end="url(#ah-on)"/><line class="path" x1="398" y1="131" x2="398" y2="172" marker-end="url(#ah-on)"/><rect class="node on" x="91" y="25" width="38" height="30" rx="7"/><text class="t-node" x="110" y="40">1</text><rect class="node on" x="379" y="25" width="38" height="30" rx="7"/><text class="t-node" x="398" y="40">5</text><rect class="node" x="667" y="25" width="38" height="30" rx="7"/><text class="t-node" x="686" y="40">9</text><rect class="node" x="91" y="100" width="38" height="30" rx="7"/><text class="t-node" x="110" y="115">1</text><rect class="node" x="235" y="100" width="38" height="30" rx="7"/><text class="t-node" x="254" y="115">3</text><rect class="node on" x="379" y="100" width="38" height="30" rx="7"/><text class="t-node" x="398" y="115">5</text><rect class="node" x="523" y="100" width="38" height="30" rx="7"/><text class="t-node" x="542" y="115">7</text><rect class="node" x="667" y="100" width="38" height="30" rx="7"/><text class="t-node" x="686" y="115">9</text><rect class="node" x="91" y="175" width="38" height="30" rx="7"/><text class="t-node" x="110" y="190">1</text><rect class="node" x="163" y="175" width="38" height="30" rx="7"/><text class="t-node" x="182" y="190">2</text><rect class="node" x="235" y="175" width="38" height="30" rx="7"/><text class="t-node" x="254" y="190">3</text><rect class="node" x="307" y="175" width="38" height="30" rx="7"/><text class="t-node" x="326" y="190">4</text><rect class="node on" x="379" y="175" width="38" height="30" rx="7"/><text class="t-node" x="398" y="190">5</text><rect class="node hit" x="451" y="175" width="38" height="30" rx="7"/><text class="t-node" x="470" y="190">6</text><rect class="node" x="523" y="175" width="38" height="30" rx="7"/><text class="t-node" x="542" y="190">7</text><rect class="node" x="595" y="175" width="38" height="30" rx="7"/><text class="t-node" x="614" y="190">8</text><rect class="node" x="667" y="175" width="38" height="30" rx="7"/><text class="t-node" x="686" y="190">9</text></svg><figcaption>Searching a skip list for 6: one long hop on level 2, two drops, one short step on level 0.</figcaption></figure>

To find a key, start at the top-left and move right while the next key is not past your target. When it would overshoot, drop one level and continue. The top levels cover most of the distance in a few long jumps; the bottom level does the final precise steps. Search and insertion both take $$O(\log N)$$ expected time, and nothing is ever rebalanced: each element's height is decided once, by a coin flip, when it is inserted.

Keep two properties in mind, because HNSW copies both: **long links on sparse upper levels, short links on the dense bottom level**, and **a random height per element**.

## Idea 2: navigable small world graphs

A skip list works for one-dimensional keys, which have an order. Vectors have no order, only distances. The graph answer is a **proximity graph**: every vector is a vertex, linked to a handful of vectors near it (its *friends*). To search, walk it greedily:

1. Start at an entry vertex.
2. Look at the current vertex's friends. Move to whichever one is closest to the query.
3. Stop when no friend is closer than where you are. That vertex is your answer.

<figure class="fig"><svg viewBox="0 0 720 310" role="img" aria-label="Greedy search on a proximity graph: from the entry point A it jumps along a long-range link to F, then to I, then to J, the nearest vertex to the query, where no neighbour is closer."><defs><marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path class="arrow" d="M0,0L10,5L0,10z"/></marker><marker id="ah-on" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path class="arrow-on" d="M0,0L10,5L0,10z"/></marker></defs><line class="edge" x1="80" y1="230" x2="170" y2="150"/><line class="edge" x1="80" y1="230" x2="150" y2="265"/><line class="edge" x1="80" y1="230" x2="260" y2="210"/><line class="edge" x1="170" y1="150" x2="290" y2="90"/><line class="edge" x1="170" y1="150" x2="260" y2="210"/><line class="edge" x1="150" y1="265" x2="260" y2="210"/><line class="edge" x1="260" y1="210" x2="380" y2="170"/><line class="edge" x1="260" y1="210" x2="400" y2="265"/><line class="edge" x1="290" y1="90" x2="380" y2="170"/><line class="edge" x1="290" y1="90" x2="470" y2="95"/><line class="edge" x1="380" y1="170" x2="470" y2="95"/><line class="edge" x1="400" y1="265" x2="540" y2="200"/><line class="edge" x1="470" y1="95" x2="560" y2="50"/><line class="edge" x1="470" y1="95" x2="540" y2="200"/><line class="edge" x1="540" y1="200" x2="645" y2="250"/><line class="edge" x1="620" y1="130" x2="560" y2="50"/><line class="edge" x1="620" y1="130" x2="645" y2="250"/><line class="long" x1="150" y1="265" x2="400" y2="265"/><line class="long" x1="290" y1="90" x2="560" y2="50"/><line class="path" x1="95.7" y1="226.9" x2="361.4" y2="173.7" marker-end="url(#ah-on)"/><line class="path" x1="395.7" y1="172.9" x2="521.3" y2="196.5" marker-end="url(#ah-on)"/><line class="path" x1="552.0" y1="189.5" x2="605.7" y2="142.5" marker-end="url(#ah-on)"/><circle class="node entry" cx="80" cy="230" r="15"/><text class="t-node" x="80" y="230">A</text><circle class="node" cx="170" cy="150" r="15"/><text class="t-node" x="170" y="150">B</text><circle class="node" cx="150" cy="265" r="15"/><text class="t-node" x="150" y="265">C</text><circle class="node" cx="260" cy="210" r="15"/><text class="t-node" x="260" y="210">D</text><circle class="node" cx="290" cy="90" r="15"/><text class="t-node" x="290" y="90">E</text><circle class="node on" cx="380" cy="170" r="15"/><text class="t-node" x="380" y="170">F</text><circle class="node" cx="400" cy="265" r="15"/><text class="t-node" x="400" y="265">G</text><circle class="node" cx="470" cy="95" r="15"/><text class="t-node" x="470" y="95">H</text><circle class="node on" cx="540" cy="200" r="15"/><text class="t-node" x="540" y="200">I</text><circle class="node hit" cx="620" cy="130" r="15"/><text class="t-node" x="620" y="130">J</text><circle class="node" cx="645" cy="250" r="15"/><text class="t-node" x="645" y="250">K</text><circle class="node" cx="560" cy="50" r="15"/><text class="t-node" x="560" y="50">L</text><polygon class="query" points="600.0,164.0 602.6,171.4 610.5,171.6 604.3,176.4 606.5,183.9 600.0,179.5 593.5,183.9 595.7,176.4 589.5,171.6 597.4,171.4"/><text class="t-note" x="586" y="205">query</text><text class="t-note" x="52" y="298">entry point</text><line class="long" x1="470" y1="292" x2="510" y2="292"/><text class="t-note" x="518" y="297">long-range link</text></svg><figcaption>Greedy search on a navigable small world graph. From entry point A, a long-range link reaches F in one hop; short links close in through I to J, where no neighbour is closer to the query.</figcaption></figure>

The graph is **navigable** when this greedy walk reaches the right place in few hops, roughly logarithmic in $$N$$. What makes that possible is a mix of **short links**, which make the final approach precise, and **long-range links**, which cross the space quickly. In the figure the walk from A jumps across the graph on a long link to F in one hop, then closes in through I to J with short ones. Without long links, the same walk would crawl through every vertex in between.

Navigable Small World (NSW) graphs, introduced by Yury Malkov and colleagues in the early 2010s, get those long links for free: vectors are inserted one at a time, each linked to its nearest neighbours *among the vectors already inserted*. Early vectors are linked when the graph is sparse, so their links are long; later vectors are linked when it is dense, so theirs are short.

NSW has two weaknesses:

- **Local minima.** Greedy search stops at a vertex with no closer friend, which is not always the true nearest neighbour. More friends per vertex lowers the risk but makes every hop more expensive.
- **Hubs.** The early vertices collect huge friend lists, and the walk spends much of its time scanning them. Search cost grows polylogarithmically instead of logarithmically.

## HNSW: a skip list made of graphs

HNSW (Malkov and Yashunin, 2016) fixes both by applying the skip-list idea to NSW. Instead of one graph that mixes long and short links, it builds a **stack of graphs**:

- **Layer 0** holds every vector, linked to its close neighbours: short links, precise.
- **Each layer above** holds an exponentially smaller random subset, so its links, measured between far fewer vectors, are long.
- The **entry point** is a vector on the top layer.

<figure class="fig"><svg viewBox="0 0 720 400" role="img" aria-label="HNSW as three stacked layers. The search enters at the top layer, crosses one long link, drops down, takes a shorter step, drops to layer 0 and walks two short links to the nearest neighbour."><defs><marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path class="arrow" d="M0,0L10,5L0,10z"/></marker><marker id="ah-on" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path class="arrow-on" d="M0,0L10,5L0,10z"/></marker></defs><polygon class="plane" points="100,38 700,38 620,120 20,120"/><text class="t-label" x="104" y="30">LAYER 2 · few vectors, long links</text><polygon class="plane" points="100,168 700,168 620,250 20,250"/><text class="t-label" x="104" y="160">LAYER 1</text><polygon class="plane" points="100,298 700,298 620,380 20,380"/><text class="t-label" x="104" y="290">LAYER 0 · every vector, short links</text><line class="drop" x1="150" y1="89" x2="150" y2="327"/><line class="drop" x1="560" y1="77" x2="560" y2="315"/><line class="drop" x1="300" y1="235" x2="300" y2="343"/><line class="drop" x1="430" y1="199" x2="430" y2="307"/><line class="edge" x1="150" y1="208" x2="300" y2="224"/><line class="edge" x1="300" y1="224" x2="430" y2="188"/><line class="edge" x1="300" y1="224" x2="560" y2="196"/><line class="edge" x1="150" y1="338" x2="178" y2="362"/><line class="edge" x1="150" y1="338" x2="215" y2="320"/><line class="edge" x1="215" y1="320" x2="262" y2="362"/><line class="edge" x1="262" y1="362" x2="300" y2="354"/><line class="edge" x1="178" y1="362" x2="262" y2="362"/><line class="edge" x1="300" y1="354" x2="370" y2="358"/><line class="edge" x1="370" y1="358" x2="395" y2="332"/><line class="edge" x1="395" y1="332" x2="430" y2="318"/><line class="edge" x1="522" y1="314" x2="560" y2="326"/><line class="edge" x1="370" y1="358" x2="482" y2="354"/><line class="edge" x1="560" y1="326" x2="592" y2="350"/><line class="edge" x1="592" y1="350" x2="482" y2="354"/><line class="edge" x1="395" y1="332" x2="482" y2="354"/><line class="path" x1="162.0" y1="77.6" x2="545.0" y2="66.4" marker-end="url(#ah-on)"/><line class="path" x1="548.0" y1="195.3" x2="445.0" y2="188.9" marker-end="url(#ah-on)"/><line class="path" x1="442.0" y1="317.5" x2="507.0" y2="314.7" marker-end="url(#ah-on)"/><line class="path" x1="513.5" y1="322.5" x2="492.6" y2="343.4" marker-end="url(#ah-on)"/><line class="path" x1="560.0" y1="78.0" x2="560.0" y2="181.0" marker-end="url(#ah-on)"/><line class="path" x1="430.0" y1="200.0" x2="430.0" y2="303.0" marker-end="url(#ah-on)"/><circle class="node entry" cx="150" cy="78" r="10"/><circle class="node on" cx="560" cy="66" r="10"/><circle class="node" cx="150" cy="208" r="10"/><circle class="node on" cx="560" cy="196" r="10"/><circle class="node" cx="300" cy="224" r="10"/><circle class="node on" cx="430" cy="188" r="10"/><circle class="node" cx="150" cy="338" r="10"/><circle class="node" cx="560" cy="326" r="10"/><circle class="node" cx="300" cy="354" r="10"/><circle class="node on" cx="430" cy="318" r="10"/><circle class="node" cx="215" cy="320" r="10"/><circle class="node" cx="262" cy="362" r="10"/><circle class="node" cx="370" cy="358" r="10"/><circle class="node hit" cx="482" cy="354" r="10"/><circle class="node on" cx="522" cy="314" r="10"/><circle class="node" cx="395" cy="332" r="10"/><circle class="node" cx="178" cy="362" r="10"/><circle class="node" cx="592" cy="350" r="10"/><polygon class="query" points="500.0,341.0 502.2,346.9 508.6,347.2 503.6,351.2 505.3,357.3 500.0,353.8 494.7,357.3 496.4,351.2 491.4,347.2 497.8,346.9"/><text class="t-note" x="36" y="394">entry point</text><circle class="node entry" cx="24" cy="389" r="6"/><polygon class="query" points="140.0,382.0 141.8,386.6 146.7,386.8 142.9,389.9 144.1,394.7 140.0,392.0 135.9,394.7 137.1,389.9 133.3,386.8 138.2,386.6"/><text class="t-note" x="150" y="394">query</text><circle class="node hit" cx="224" cy="389" r="6"/><text class="t-note" x="236" y="394">nearest neighbour found</text></svg><figcaption>HNSW search: one long hop on the top layer, drop down, one shorter hop, drop to layer 0 and finish with short steps to the nearest neighbour.</figcaption></figure>

Search walks the stack top down. On each upper layer it runs the greedy walk to a local minimum using only the closest vertex (a beam of width 1), then drops that vertex down one layer and starts again from it. On layer 0 it widens the beam to **efSearch** candidates and returns the best $$k$$.

The top layers do the long-distance travel in a few hops over very few vertices; layer 0 does the local refinement. Because the number of vertices shrinks geometrically going up, the expected number of hops is logarithmic in $$N$$, and hubs never form: each layer's links are chosen among vectors at that layer's scale.

### Searching one layer

The core routine is a best-first search that keeps two heaps: **candidates** still to expand (closest first) and the **results** found so far (worst first, capped at `ef`).

```python
import heapq

def search_layer(q, entry_points, ef, graph, dist):
    """Best-first search on one layer. Returns up to ef (distance, node) pairs, closest first."""
    visited = set(entry_points)
    candidates = [(dist(q, e), e) for e in entry_points]  # min-heap: closest first
    heapq.heapify(candidates)
    results = [(-d, e) for d, e in candidates]            # max-heap: farthest first
    heapq.heapify(results)

    while candidates:
        d, c = heapq.heappop(candidates)
        if d > -results[0][0]:        # the best unexpanded candidate is worse than our worst result:
            break                     # nothing left can improve the answer
        for n in graph[c]:            # c's neighbour list on this layer
            if n in visited:
                continue
            visited.add(n)
            dn = dist(q, n)
            if len(results) < ef or dn < -results[0][0]:
                heapq.heappush(candidates, (dn, n))
                heapq.heappush(results, (-dn, n))
                if len(results) > ef:
                    heapq.heappop(results)  # drop the current worst
    return sorted((-d, n) for d, n in results)


def search(q, index, k, ef_search):
    ep = [index.entry_point]
    for layer in range(index.max_layer, 0, -1):   # upper layers: beam of 1, just descend
        ep = [search_layer(q, ep, 1, index.graphs[layer], index.dist)[0][1]]
    hits = search_layer(q, ep, max(ef_search, k), index.graphs[0], index.dist)
    return hits[:k]
```

`ef` is the beam width. With `ef = 1` this is exactly the greedy walk from the figure. A larger `ef` keeps more alternatives alive, so the search can escape a local minimum through a slightly worse vertex that leads somewhere better. That single number is the recall knob you turn at query time.

## Building the graph

Vectors are inserted one at a time. Each insertion does three things.

**1. Pick the vector's top layer at random.** Like the coin flips in a skip list, the level is drawn from an exponentially decaying distribution:

$$
\ell = \left\lfloor -\ln(U) \cdot m_L \right\rfloor, \qquad U \sim \text{Uniform}(0, 1)
$$

The vector then lives on layer $$\ell$$ and every layer below it. This gives $$P(\ell \ge l) = e^{-l / m_L}$$. The paper's recommended normalisation is $$m_L = 1 / \ln M$$, which makes each layer hold $$1/M$$ of the layer below. With $$M = 32$$, only 1 vector in 32 reaches layer 1, 1 in 1,024 reaches layer 2, and so on.

It really is that simple. Here is the level distribution Faiss produced when building a 200,000-vector index with $$M = 32$$, next to what the formula predicts:

| Layer | Probability of that exact layer | Expected count | Measured count |
|---|---|---|---|
| 0 | 0.96875 | 193,750 | 193,725 |
| 1 | 0.03027 | 6,055 | 6,088 |
| 2 | 0.00095 | 189 | 182 |
| 3 | 0.00003 | 6 | 4 |
| 4 | 0.000001 | 0.2 | 1 |

The single vector on layer 4 is the entry point. Five layers are enough for 200,000 vectors, and a billion would need only about six.

**2. Find the insertion point.** Starting from the entry point, descend through the layers above $$\ell$$ with a beam of 1, exactly like a search. From layer $$\ell$$ down to layer 0, switch to a beam of **efConstruction** to collect a good list of candidate neighbours on each layer.

**3. Choose the neighbours and link both ways.** From the candidates, pick up to **M** neighbours and add bidirectional links. If a neighbour now has too many links (more than $$M_{max} = M$$ on upper layers, or $$M_{max0} = 2M$$ on layer 0), prune its list back down.

Layer 0 gets twice the links because it is where the precise search happens and where every vector lives.

### The neighbour-selection heuristic

Picking the $$M$$ closest candidates sounds right and works badly on clustered data: all $$M$$ links land inside the same tight cluster, and the graph loses the links that lead out of it. HNSW uses a diversity rule instead. Keep a candidate only if it is closer to the new vector than to every neighbour already kept:

```python
def select_neighbors(q, candidates, M, dist):
    """candidates: (distance to q, node) pairs. Prefer neighbours in different directions."""
    kept = []
    for d_q, c in sorted(candidates):
        if all(d_q < dist(c, other) for other in kept):
            kept.append(c)
        if len(kept) == M:
            break
    return kept
```

A candidate that sits behind an already-kept neighbour is reachable through that neighbour anyway, so a link to it is wasted. Skipping it spends the link budget on other directions, including the bridges between clusters that greedy search depends on.

## What the three knobs cost

HNSW has three parameters. Two are fixed when you build the index; one you choose per query.

| Parameter | Set when | Controls | Costs |
|---|---|---|---|
| **M** | build | links per vector (2M on layer 0) | memory, build time, time per hop |
| **efConstruction** | build | beam width while inserting, so graph quality | build time only |
| **efSearch** | query | beam width while searching | query latency |

To see the trade-offs as numbers, I built Faiss `IndexHNSWFlat` indexes over **200,000 vectors of 128 dimensions** and searched them with **1,000 held-out queries**, measuring recall@10 against exact brute-force results. The vectors are synthetic but shaped like real embeddings: points on a 24-dimensional subspace embedded in 128 dimensions, plus a little noise. Build times use 8 threads; query times use a single thread on an Apple M5 Pro. Your data will give different absolute numbers; the shapes are what transfer.

### efSearch: the query-time dial

Recall@10, and milliseconds per query, with efConstruction = 128:

| efSearch | M = 8 | M = 16 | M = 32 | M = 64 |
|---|---|---|---|---|
| 10 | 0.318 · 0.013 ms | 0.533 · 0.020 ms | 0.664 · 0.028 ms | 0.721 · 0.037 ms |
| 16 | 0.411 · 0.016 ms | 0.651 · 0.026 ms | 0.774 · 0.038 ms | 0.822 · 0.045 ms |
| 32 | 0.575 · 0.029 ms | 0.810 · 0.043 ms | 0.900 · 0.063 ms | 0.927 · 0.077 ms |
| 64 | 0.734 · 0.052 ms | 0.926 · 0.082 ms | 0.974 · 0.120 ms | 0.984 · 0.142 ms |
| 128 | 0.855 · 0.104 ms | 0.979 · 0.163 ms | 0.996 · 0.228 ms | 0.998 · 0.268 ms |
| 256 | 0.931 · 0.223 ms | 0.996 · 0.321 ms | 0.999 · 0.434 ms | 1.000 · 0.505 ms |

Three things to read off this table:

- **Latency is roughly linear in efSearch.** Doubling the beam roughly doubles the work, at every M.
- **Recall saturates.** At M = 16, going from 64 to 128 buys 5 points; going from 128 to 256 buys under 2 for twice the cost.
- **Compare at equal latency, not equal efSearch.** Higher M looks far better per efSearch value, but each hop scans more links. Around 0.08 ms, M = 16 at efSearch 64 (0.926) and M = 64 at efSearch 32 (0.927) are the same. Higher M earns its keep at the high-recall end, and M = 8 never catches up at all: the graph is too sparse to escape local minima.

For scale: exact search over the same 200,000 vectors, one query at a time on one thread, took **1.39 ms**. HNSW at M = 16, efSearch = 64 took **0.085 ms** at 0.93 recall, 16 times faster, and brute force grows linearly with $$N$$ while HNSW grows roughly logarithmically.

### efConstruction: pay once, at build time

With M = 16 and efSearch fixed at 32:

| efConstruction | Build time | Recall@10 | Query time |
|---|---|---|---|
| 16 | 0.76 s | 0.635 | 0.042 ms |
| 32 | 1.04 s | 0.687 | 0.040 ms |
| 64 | 2.04 s | 0.776 | 0.045 ms |
| 128 | 3.90 s | 0.810 | 0.047 ms |
| 256 | 7.37 s | 0.814 | 0.046 ms |

A better-built graph gives better recall at **the same query cost**: 18 points between 16 and 128, for free at query time. Then it plateaus. Doubling efConstruction from 128 to 256 doubled the build time for less than half a point. Around 100 to 200 is where most production defaults sit, and this curve shows why.

### Memory: the vectors, not the graph

Measured index size, against 102.4 MB of raw float32 vectors:

| M | Index size | Bytes per vector | Links only |
|---|---|---|---|
| 8 | 118.5 MB | 593 | 64 |
| 16 | 131.3 MB | 656 | 128 |
| 32 | 156.8 MB | 784 | 256 |
| 64 | 208.0 MB | 1,040 | 512 |

The pattern is exact enough to use as a formula. Per vector, an HNSW index stores

$$
\underbrace{4d}_{\text{the vector}} \;+\; \underbrace{4 \cdot 2M}_{\text{layer-0 links}} \;+\; \approx 16 \text{ bytes of bookkeeping and upper layers}
$$

With 128 dimensions and M = 16 the links are a fifth of the index. With the 1,536-dimensional embeddings typical of modern embedding models, the vector alone is 6,144 bytes and the M = 16 links are 128: **about 2% of the total**. At that point M barely matters for memory. What matters is storing the vectors themselves compactly, which is why production systems pair HNSW with scalar or product quantisation, or keep the full vectors on disk and only the graph in RAM.

## A tuning recipe

1. **Start at M = 16, efConstruction = 128.** It is a strong default for most embedding workloads.
2. **Build a ground-truth set.** Take about 1,000 real queries and compute their exact top-k once with brute force. Without this you are tuning blind.
3. **Sweep efSearch** upwards from $$k$$ and pick the smallest value that meets your recall target. This needs no rebuild, so it is cheap to revisit.
4. **If the target needs efSearch in the hundreds,** rebuild with a higher M (32, then 48 or 64) and sweep again at equal latency, not equal efSearch.
5. **Raise efConstruction only until recall stops moving.** It is paid at build time and for every re-index, so there is no point overshooting the plateau.
6. **Plan memory from the formula,** and quantise the vectors if $$4d$$ dominates.

## Where HNSW bites

- **Recall decays as the index grows.** Settings that give 0.95 at a million vectors can give less at a hundred million. Re-measure after large ingests, not just at launch.
- **Filters fragment the graph.** A restrictive metadata filter removes most vertices from the walk, and the remaining ones may not be connected. Engines handle this with filter-aware traversal, payload indexes or a fallback to exact search for small candidate sets; know which one yours uses.
- **Deletes are tombstones.** Removing a vector from a graph is hard, so most implementations mark it deleted and skip it. Heavy churn slowly degrades the graph until you rebuild or compact.
- **efSearch must be at least k.** Most libraries silently raise it, but a beam narrower than the number of results you want cannot return them.
- **It all lives in RAM.** The random access pattern of a graph walk is why HNSW is fast and also why it is expensive at scale.

## Summary

HNSW is a skip list whose levels are proximity graphs. Sparse upper layers with long links get the search to the right neighbourhood in a few hops; the dense bottom layer, with a wider beam, finds the precise answer. Each vector's height is a single random draw, and each vector's links are chosen for diversity rather than raw closeness.

That structure explains the knobs. **M** is graph density: recall per hop against memory and time per hop. **efConstruction** is graph quality, paid once and plateauing quickly. **efSearch** is the beam you trade for latency on every query, and the one you should tune against a ground-truth set.

For HNSW implemented from scratch in Python, with parameter sweeps, recall degradation and filtered search measured, see [Chapter 22 of the RAG book](../books/rag-first-principles/part8-deep-dives/22-vector-search-internals.md).

## References

1. Y. Malkov, D. Yashunin. *Efficient and robust approximate nearest neighbor search using Hierarchical Navigable Small World graphs.* IEEE Transactions on Pattern Analysis and Machine Intelligence, 2018 ([arXiv:1603.09320](https://arxiv.org/abs/1603.09320)).
2. W. Pugh. *Skip lists: a probabilistic alternative to balanced trees.* Communications of the ACM, 33(6), 1990.
3. Y. Malkov, A. Ponomarenko, A. Logvinov, V. Krylov. *Approximate nearest neighbor algorithm based on navigable small world graphs.* Information Systems, 45, 2014.
4. M. Aumüller, E. Bernhardsson, A. Faithfull. *ANN-Benchmarks: a benchmarking tool for approximate nearest neighbor algorithms.* Information Systems, 87, 2020 ([ann-benchmarks.com](https://ann-benchmarks.com)).
5. [Faiss](https://github.com/facebookresearch/faiss), the library used for the measurements above.

<details>
<summary>The benchmark script</summary>

```python
"""HNSW measurements: recall@10 and latency over M, efSearch and efConstruction."""
import time
import numpy as np
import faiss

rng = np.random.default_rng(7)
N, NQ, D, LATENT, K = 200_000, 1_000, 128, 24, 10

# Synthetic vectors with low intrinsic dimension, like real embeddings.
proj = rng.standard_normal((LATENT, D)).astype("float32")
def sample(n):
    z = rng.standard_normal((n, LATENT)).astype("float32")
    return z @ proj + 0.1 * rng.standard_normal((n, D)).astype("float32")

xb, xq = sample(N), sample(NQ)
flat = faiss.IndexFlatL2(D)
flat.add(xb)
_, gt = flat.search(xq, K)                     # exact ground truth

def recall(I):
    return np.mean([len(set(I[i]) & set(gt[i])) / K for i in range(NQ)])

def build(M, ef_construction):
    faiss.omp_set_num_threads(8)
    index = faiss.IndexHNSWFlat(D, M)
    index.hnsw.efConstruction = ef_construction
    t = time.perf_counter(); index.add(xb)
    return index, time.perf_counter() - t

def search(index, ef_search):
    faiss.omp_set_num_threads(1)
    index.hnsw.efSearch = ef_search
    t = time.perf_counter(); _, I = index.search(xq, K)
    return recall(I), (time.perf_counter() - t) * 1000 / NQ   # recall, ms per query

for M in (8, 16, 32, 64):
    index, secs = build(M, 128)
    size_mb = len(faiss.serialize_index(index)) / 1e6
    print(f"M={M}: built in {secs:.1f}s, {size_mb:.1f} MB")
    for ef in (10, 16, 32, 64, 128, 256):
        r, ms = search(index, ef)
        print(f"  efSearch={ef:<4} recall@10={r:.3f}  {ms:.3f} ms/query")

levels = faiss.vector_to_array(index.hnsw.levels) - 1   # Faiss stores level + 1
print("vectors per layer:", np.bincount(levels))
```

</details>
