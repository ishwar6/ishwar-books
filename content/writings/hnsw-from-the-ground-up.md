---
title: "HNSW, From the Ground Up"
description: "How Hierarchical Navigable Small World graphs find nearest neighbours in microseconds: the two ideas they combine, how the graph is built, what M, efConstruction and efSearch really cost, and measured numbers from a 200,000-vector index."
date: 2026-09-27
tags: [vector-search, hnsw, retrieval]
motif: graph
accent: "#7c9cff"
---

**HNSW** (Hierarchical Navigable Small World graphs) is one of the most widely used indexes for finding similar vectors. Qdrant, Weaviate, pgvector and Elasticsearch all offer it, and several use it by default. It earns that place. On the index measured in this piece it returns 93% of the true top 10 in **0.085 ms**, where checking every vector takes **1.39 ms**, and the gap widens with every vector you add.

It is also an algorithm most people use as a black box with three knobs. This piece opens the box. HNSW is two older ideas glued together, and once you see both, every parameter stops being magic and becomes a trade you can reason about.

> [!TIP] The whole idea in one minute
> Think of travelling to a friend's house in another country. You take a **flight** to the right city, a **train** to the right neighbourhood, then **walk** the last few streets. Each stage is finer than the one before, and gets you close enough for the next one to finish the job.
>
> HNSW stores vectors the same way: a few vectors on the top layer with long links (the flights), more on each layer below, and every vector on the bottom layer with short links (the streets). A search starts at the top, hops towards the query, drops a layer, and repeats until it is walking between close neighbours on the bottom layer.

## The problem: exact search does not scale

Given a query vector, find the $$k$$ stored vectors closest to it. The exact answer means computing the distance to every one of the $$N$$ stored vectors, and each distance touches all $$d$$ coordinates: $$O(N \cdot d)$$ work per query. At 200,000 vectors of 128 dimensions that is 25.6 million coordinates to compare for every single query (for Euclidean distance, a subtract, a multiply and an add each). At 100 million vectors it is hopeless for anything interactive.

**Approximate** nearest-neighbour (ANN) search gives up a little accuracy for a lot of speed. Its quality is measured as **recall@k**: of the true $$k$$ nearest neighbours, what fraction did the index return? Recall 0.95 at $$k = 10$$ means that, on average, 9.5 of the 10 results are the true top 10.

ANN methods come in a few families:

| Family | Idea | Examples |
|---|---|---|
| Trees | recursively split the space | k-d trees, Annoy |
| Hashing | hash similar vectors into the same bucket | LSH |
| Partitioning | cluster the vectors, then search only the clusters nearest the query | IVF |
| Quantisation | compress each vector so distances are cheaper to compute | scalar (SQ), product (PQ), binary |
| **Graphs** | connect each vector to its neighbours, then walk the graph | NSW, **HNSW**, Vamana |

These families solve different problems and are often **combined**. Partitioning decides *which* vectors to look at; quantisation makes looking at each one cheaper. Faiss's popular IVF-PQ index is both: IVF picks the clusters, PQ compresses the vectors inside them. HNSW is also frequently paired with quantisation, as the memory section below explains.

Trees break down in high dimensions and hashing needs many tables to reach high recall. Graphs have led public ANN benchmarks for years, and HNSW is the graph method most systems ship.

## A detour through sorted data

Before vectors, start with something easier: a sorted list of numbers.

```
3   5   7   11   14   19   21
```

How fast can we find 19, and how cheaply can we insert 13? Three data structures give three different answers.

**A sorted array** supports **binary search**. Look at the middle element (11). 19 is bigger, so throw away the left half and repeat on `14 19 21`. The middle of that is 19: found. Each step halves what is left, so a search takes $$O(\log n)$$ steps. A million sorted numbers need about 20 comparisons.

**But inserting into an array is slow.** To put 13 between 11 and 14, every element after it has to shift one place to the right. That is $$O(n)$$ work per insert.

**A linked list** is the opposite. Each element points to the next, so inserting 13 means changing two pointers: 11 now points to 13, and 13 points to 14. That is $$O(1)$$, once you know where to insert. The catch is *finding* that place: a linked list has no middle you can jump to, so you walk from the start, one element at a time. Search is $$O(n)$$.

<figure class="fig"><svg viewBox="0 0 720 300" role="img" aria-label="Inserting 13. In a sorted array, 14, 19 and 21 each shift one place right. In a linked list, only two pointers change: 11 now points to 13, and 13 points to 14."><defs><marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path class="arrow" d="M0,0L10,5L0,10z"/></marker><marker id="ah-on" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path class="arrow-on" d="M0,0L10,5L0,10z"/></marker><marker id="ah-bad" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path class="arrow-bad" d="M0,0L10,5L0,10z"/></marker></defs><text class="t-label" x="20" y="22">SORTED ARRAY</text><text class="t-note" x="20" y="67">before</text><rect class="node" x="150" y="45" width="50" height="34" rx="4"/><text class="t-node" x="175" y="62">3</text><rect class="node" x="204" y="45" width="50" height="34" rx="4"/><text class="t-node" x="229" y="62">5</text><rect class="node" x="258" y="45" width="50" height="34" rx="4"/><text class="t-node" x="283" y="62">7</text><rect class="node" x="312" y="45" width="50" height="34" rx="4"/><text class="t-node" x="337" y="62">11</text><rect class="node" x="366" y="45" width="50" height="34" rx="4"/><text class="t-node" x="391" y="62">14</text><rect class="node" x="420" y="45" width="50" height="34" rx="4"/><text class="t-node" x="445" y="62">19</text><rect class="node" x="474" y="45" width="50" height="34" rx="4"/><text class="t-node" x="499" y="62">21</text><text class="t-note" x="20" y="137">after</text><rect class="node" x="150" y="115" width="50" height="34" rx="4"/><text class="t-node" x="175" y="132">3</text><rect class="node" x="204" y="115" width="50" height="34" rx="4"/><text class="t-node" x="229" y="132">5</text><rect class="node" x="258" y="115" width="50" height="34" rx="4"/><text class="t-node" x="283" y="132">7</text><rect class="node" x="312" y="115" width="50" height="34" rx="4"/><text class="t-node" x="337" y="132">11</text><rect class="node hit" x="366" y="115" width="50" height="34" rx="4"/><text class="t-node" x="391" y="132">13</text><rect class="node on" x="420" y="115" width="50" height="34" rx="4"/><text class="t-node" x="445" y="132">14</text><rect class="node on" x="474" y="115" width="50" height="34" rx="4"/><text class="t-node" x="499" y="132">19</text><rect class="node on" x="528" y="115" width="50" height="34" rx="4"/><text class="t-node" x="553" y="132">21</text><line class="path" x1="391" y1="81" x2="439" y2="113" marker-end="url(#ah-on)"/><line class="path" x1="445" y1="81" x2="493" y2="113" marker-end="url(#ah-on)"/><line class="path" x1="499" y1="81" x2="547" y2="113" marker-end="url(#ah-on)"/><text class="t-note" x="593" y="92">every later element</text><text class="t-note" x="593" y="110">shifts: O(n)</text><text class="t-label" x="20" y="186">LINKED LIST</text><line class="edge" x1="120" y1="222" x2="166" y2="222" marker-end="url(#ah)"/><line class="edge" x1="208" y1="222" x2="254" y2="222" marker-end="url(#ah)"/><line class="edge" x1="296" y1="222" x2="342" y2="222" marker-end="url(#ah)"/><line class="long" x1="384" y1="222" x2="430" y2="222"/><line class="edge" x1="472" y1="222" x2="518" y2="222" marker-end="url(#ah)"/><line class="edge" x1="560" y1="222" x2="606" y2="222" marker-end="url(#ah)"/><line class="path" x1="376" y1="238" x2="386.0" y2="262" marker-end="url(#ah-on)"/><line class="path" x1="430.0" y1="262" x2="440" y2="240" marker-end="url(#ah-on)"/><circle class="node" cx="100" cy="222" r="18"/><text class="t-node" x="100" y="222">3</text><circle class="node" cx="188" cy="222" r="18"/><text class="t-node" x="188" y="222">5</text><circle class="node" cx="276" cy="222" r="18"/><text class="t-node" x="276" y="222">7</text><circle class="node on" cx="364" cy="222" r="18"/><text class="t-node" x="364" y="222">11</text><circle class="node on" cx="452" cy="222" r="18"/><text class="t-node" x="452" y="222">14</text><circle class="node" cx="540" cy="222" r="18"/><text class="t-node" x="540" y="222">19</text><circle class="node" cx="628" cy="222" r="18"/><text class="t-node" x="628" y="222">21</text><circle class="node hit" cx="408.0" cy="272" r="18"/><text class="t-node" x="408.0" y="272">13</text><text class="t-note" x="520" y="277">rewire two pointers: O(1)</text></svg><figcaption>Inserting 13. The sorted array shifts every later element; the linked list rewires two pointers, but had to walk from the start to find 11.</figcaption></figure>

| | Search | Insert |
|---|---|---|
| Sorted array (binary search) | fast, $$O(\log n)$$ | slow, $$O(n)$$ |
| Linked list | slow, $$O(n)$$ | fast, $$O(1)$$ once found |
| **Skip list** | fast, $$O(\log n)$$ expected | fast, $$O(\log n)$$ expected |

So if your data is sorted and rarely changes, a plain array with binary search is hard to beat. The skip list exists for data that is both searched and changed often: it keeps the cheap pointer updates of a linked list and adds a way to jump.

## Idea 1: the skip list

A **skip list** (William Pugh, 1990) is a sorted linked list with express lanes. The bottom level holds every element in order. Each level above holds a random subset of the level below (typically half), so the top levels have few elements and links that jump far.

<figure class="fig"><svg viewBox="0 0 740 230" role="img" aria-label="A skip list with three levels; searching for 6 skips from 1 to 5 on the top level, drops down twice, then steps from 5 to 6 on the bottom level."><defs><marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path class="arrow" d="M0,0L10,5L0,10z"/></marker><marker id="ah-on" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path class="arrow-on" d="M0,0L10,5L0,10z"/></marker><marker id="ah-bad" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path class="arrow-bad" d="M0,0L10,5L0,10z"/></marker></defs><text class="t-label" x="8" y="45">LEVEL 2</text><text class="t-label" x="8" y="120">LEVEL 1</text><text class="t-label" x="8" y="195">LEVEL 0</text><line class="drop" x1="110" y1="56" x2="110" y2="174"/><line class="drop" x1="254" y1="131" x2="254" y2="174"/><line class="drop" x1="398" y1="56" x2="398" y2="174"/><line class="drop" x1="542" y1="131" x2="542" y2="174"/><line class="drop" x1="686" y1="56" x2="686" y2="174"/><line class="path" x1="130" y1="40" x2="376" y2="40" marker-end="url(#ah-on)"/><line class="edge" x1="418" y1="40" x2="664" y2="40" marker-end="url(#ah)"/><line class="edge" x1="130" y1="115" x2="232" y2="115" marker-end="url(#ah)"/><line class="edge" x1="274" y1="115" x2="376" y2="115" marker-end="url(#ah)"/><line class="edge" x1="418" y1="115" x2="520" y2="115" marker-end="url(#ah)"/><line class="edge" x1="562" y1="115" x2="664" y2="115" marker-end="url(#ah)"/><line class="edge" x1="130" y1="190" x2="160" y2="190" marker-end="url(#ah)"/><line class="edge" x1="202" y1="190" x2="232" y2="190" marker-end="url(#ah)"/><line class="edge" x1="274" y1="190" x2="304" y2="190" marker-end="url(#ah)"/><line class="edge" x1="346" y1="190" x2="376" y2="190" marker-end="url(#ah)"/><line class="path" x1="418" y1="190" x2="448" y2="190" marker-end="url(#ah-on)"/><line class="edge" x1="490" y1="190" x2="520" y2="190" marker-end="url(#ah)"/><line class="edge" x1="562" y1="190" x2="592" y2="190" marker-end="url(#ah)"/><line class="edge" x1="634" y1="190" x2="664" y2="190" marker-end="url(#ah)"/><line class="path" x1="398" y1="56" x2="398" y2="97" marker-end="url(#ah-on)"/><line class="path" x1="398" y1="131" x2="398" y2="172" marker-end="url(#ah-on)"/><rect class="node on" x="91" y="25" width="38" height="30" rx="7"/><text class="t-node" x="110" y="40">1</text><rect class="node on" x="379" y="25" width="38" height="30" rx="7"/><text class="t-node" x="398" y="40">5</text><rect class="node" x="667" y="25" width="38" height="30" rx="7"/><text class="t-node" x="686" y="40">9</text><rect class="node" x="91" y="100" width="38" height="30" rx="7"/><text class="t-node" x="110" y="115">1</text><rect class="node" x="235" y="100" width="38" height="30" rx="7"/><text class="t-node" x="254" y="115">3</text><rect class="node on" x="379" y="100" width="38" height="30" rx="7"/><text class="t-node" x="398" y="115">5</text><rect class="node" x="523" y="100" width="38" height="30" rx="7"/><text class="t-node" x="542" y="115">7</text><rect class="node" x="667" y="100" width="38" height="30" rx="7"/><text class="t-node" x="686" y="115">9</text><rect class="node" x="91" y="175" width="38" height="30" rx="7"/><text class="t-node" x="110" y="190">1</text><rect class="node" x="163" y="175" width="38" height="30" rx="7"/><text class="t-node" x="182" y="190">2</text><rect class="node" x="235" y="175" width="38" height="30" rx="7"/><text class="t-node" x="254" y="190">3</text><rect class="node" x="307" y="175" width="38" height="30" rx="7"/><text class="t-node" x="326" y="190">4</text><rect class="node on" x="379" y="175" width="38" height="30" rx="7"/><text class="t-node" x="398" y="190">5</text><rect class="node hit" x="451" y="175" width="38" height="30" rx="7"/><text class="t-node" x="470" y="190">6</text><rect class="node" x="523" y="175" width="38" height="30" rx="7"/><text class="t-node" x="542" y="190">7</text><rect class="node" x="595" y="175" width="38" height="30" rx="7"/><text class="t-node" x="614" y="190">8</text><rect class="node" x="667" y="175" width="38" height="30" rx="7"/><text class="t-node" x="686" y="190">9</text></svg><figcaption>Searching a skip list for 6: one long hop on level 2, two drops, one short step on level 0.</figcaption></figure>

To find a key, start at the top-left and move right while the next key is not past your target. When it would overshoot, drop one level and continue. The top levels cover most of the distance in a few long jumps; the bottom level does the final precise steps. Search and insertion both take $$O(\log N)$$ expected time, and nothing is ever rebalanced: each element's height is decided once, by a coin flip, when it is inserted.

Keep two properties in mind, because HNSW copies both: **long links on sparse upper levels, short links on the dense bottom level**, and **a random height per element**.

### Is a skip list only for sorted data?

Yes. Every step of the search is a comparison: "is the next key bigger than my target?" If it is, moving right would overshoot, so you drop down. That question only makes sense when the keys have an order, $$3 < 5 < 7 < 11 < \dots$$. Binary search needs the same thing.

**Vectors have no such order.** Take three embeddings:

```
A = [0.12, 0.91, ..., 0.32]
B = [0.88, 0.04, ..., 0.51]
C = [0.42, 0.72, ..., 0.18]
```

There is no useful way to say $$A < B < C$$. You could sort them by their first coordinate, or by their length, but two vectors that end up side by side in that order can still point in completely different directions in the other 127 dimensions. What vectors do have is **distance**: $$\text{dist}(A, C) = 0.2$$ is small, $$\text{dist}(A, B) = 0.8$$ is large. Nearest-neighbour search asks "which stored vectors are closest to my query?", not "which key is bigger or smaller?"

That is why binary search and skip lists cannot be used on vectors directly. HNSW keeps the skip list's **layers**, and replaces its **comparison** with a **distance**:

| | Skip list | HNSW |
|---|---|---|
| Data | keys with an order | vectors with a distance |
| One search step | move right while the next key is not past the target | move to whichever neighbour is closest to the query |
| Upper layers | a random subset, long jumps | a random subset, long links |
| Bottom layer | every key, in order | every vector, linked to its nearest neighbours |
| Answer | exact | approximate |

The next idea supplies the missing piece: a structure where "move closer" works the way "move right" does in a skip list.

## Idea 2: navigable small world graphs

A skip list works for one-dimensional keys, which have an order. Vectors have no order, only distances. The graph answer is a **proximity graph**: every vector is a vertex, linked to a handful of vectors near it (its *friends*). To search, walk it greedily:

1. Start at an entry vertex.
2. Look at the current vertex's friends. Move to whichever one is closest to the query.
3. Stop when no friend is closer than where you are. That vertex is your answer.

<figure class="fig"><svg viewBox="0 0 720 310" role="img" aria-label="Greedy search on a proximity graph: from the entry point A it jumps along a long-range link to F, then to I, then to J, the nearest vertex to the query, where no neighbour is closer."><defs><marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path class="arrow" d="M0,0L10,5L0,10z"/></marker><marker id="ah-on" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path class="arrow-on" d="M0,0L10,5L0,10z"/></marker><marker id="ah-bad" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path class="arrow-bad" d="M0,0L10,5L0,10z"/></marker></defs><line class="edge" x1="80" y1="230" x2="170" y2="150"/><line class="edge" x1="80" y1="230" x2="150" y2="265"/><line class="edge" x1="80" y1="230" x2="260" y2="210"/><line class="edge" x1="170" y1="150" x2="290" y2="90"/><line class="edge" x1="170" y1="150" x2="260" y2="210"/><line class="edge" x1="150" y1="265" x2="260" y2="210"/><line class="edge" x1="260" y1="210" x2="380" y2="170"/><line class="edge" x1="260" y1="210" x2="400" y2="265"/><line class="edge" x1="290" y1="90" x2="380" y2="170"/><line class="edge" x1="290" y1="90" x2="470" y2="95"/><line class="edge" x1="380" y1="170" x2="470" y2="95"/><line class="edge" x1="400" y1="265" x2="540" y2="200"/><line class="edge" x1="470" y1="95" x2="560" y2="50"/><line class="edge" x1="470" y1="95" x2="540" y2="200"/><line class="edge" x1="540" y1="200" x2="645" y2="250"/><line class="edge" x1="620" y1="130" x2="560" y2="50"/><line class="edge" x1="620" y1="130" x2="645" y2="250"/><line class="long" x1="150" y1="265" x2="400" y2="265"/><line class="long" x1="290" y1="90" x2="560" y2="50"/><line class="path" x1="95.7" y1="226.9" x2="361.4" y2="173.7" marker-end="url(#ah-on)"/><line class="path" x1="395.7" y1="172.9" x2="521.3" y2="196.5" marker-end="url(#ah-on)"/><line class="path" x1="552.0" y1="189.5" x2="605.7" y2="142.5" marker-end="url(#ah-on)"/><circle class="node entry" cx="80" cy="230" r="15"/><text class="t-node" x="80" y="230">A</text><circle class="node" cx="170" cy="150" r="15"/><text class="t-node" x="170" y="150">B</text><circle class="node" cx="150" cy="265" r="15"/><text class="t-node" x="150" y="265">C</text><circle class="node" cx="260" cy="210" r="15"/><text class="t-node" x="260" y="210">D</text><circle class="node" cx="290" cy="90" r="15"/><text class="t-node" x="290" y="90">E</text><circle class="node on" cx="380" cy="170" r="15"/><text class="t-node" x="380" y="170">F</text><circle class="node" cx="400" cy="265" r="15"/><text class="t-node" x="400" y="265">G</text><circle class="node" cx="470" cy="95" r="15"/><text class="t-node" x="470" y="95">H</text><circle class="node on" cx="540" cy="200" r="15"/><text class="t-node" x="540" y="200">I</text><circle class="node hit" cx="620" cy="130" r="15"/><text class="t-node" x="620" y="130">J</text><circle class="node" cx="645" cy="250" r="15"/><text class="t-node" x="645" y="250">K</text><circle class="node" cx="560" cy="50" r="15"/><text class="t-node" x="560" y="50">L</text><polygon class="query" points="600.0,164.0 602.6,171.4 610.5,171.6 604.3,176.4 606.5,183.9 600.0,179.5 593.5,183.9 595.7,176.4 589.5,171.6 597.4,171.4"/><text class="t-note" x="586" y="205">query</text><text class="t-note" x="52" y="298">entry point</text><line class="long" x1="470" y1="292" x2="510" y2="292"/><text class="t-note" x="518" y="297">long-range link</text></svg><figcaption>Greedy search on a navigable small world graph. From entry point A, a long-range link reaches F in one hop; short links close in through I to J, where no neighbour is closer to the query.</figcaption></figure>

The graph is **navigable** when this greedy walk reaches the right place in few hops. What makes that possible is a mix of **short links**, which make the final approach precise, and **long-range links**, which cross the space quickly. In the figure the walk from A jumps across the graph on a long link to F in one hop, then closes in through I to J with short ones. Without long links, the same walk would crawl through every vertex in between.

### Why short links alone get stuck

The obvious graph links every vector to its few nearest neighbours and nothing else. Greedy search on that graph fails in a specific way: it can walk into a **dead end**, a vertex whose neighbours are all farther from the query than it is, even though the true answer is somewhere else.

<figure class="fig"><svg viewBox="0 0 720 300" role="img" aria-label="With only short links, greedy search from E moves to a, then b, and stops: b has no neighbour closer to the query. The true nearest neighbour N is only reachable by first moving away, through c, d and e. A single long link from b to N would fix it."><defs><marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path class="arrow" d="M0,0L10,5L0,10z"/></marker><marker id="ah-on" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path class="arrow-on" d="M0,0L10,5L0,10z"/></marker><marker id="ah-bad" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path class="arrow-bad" d="M0,0L10,5L0,10z"/></marker></defs><line class="edge" x1="70" y1="140" x2="170" y2="222"/><line class="edge" x1="170" y1="222" x2="330" y2="248"/><line class="edge" x1="330" y1="248" x2="490" y2="232"/><line class="edge" x1="490" y1="232" x2="610" y2="168"/><line class="long" x1="380" y1="92" x2="610" y2="168"/><line class="path bad" x1="85.7" y1="136.9" x2="201.4" y2="113.7" marker-end="url(#ah-bad)"/><line class="path bad" x1="235.9" y1="108.2" x2="361.1" y2="94.1" marker-end="url(#ah-bad)"/><circle class="node entry" cx="70" cy="140" r="15"/><text class="t-node" x="70" y="140">E</text><circle class="node on" cx="220" cy="110" r="15"/><text class="t-node" x="220" y="110">a</text><circle class="node stuck" cx="380" cy="92" r="15"/><text class="t-node" x="380" y="92">b</text><circle class="node" cx="170" cy="222" r="15"/><text class="t-node" x="170" y="222">c</text><circle class="node" cx="330" cy="248" r="15"/><text class="t-node" x="330" y="248">d</text><circle class="node" cx="490" cy="232" r="15"/><text class="t-node" x="490" y="232">e</text><circle class="node hit" cx="610" cy="168" r="15"/><text class="t-node" x="610" y="168">N</text><polygon class="query" points="645.0,107.0 647.6,114.4 655.5,114.6 649.3,119.4 651.5,126.9 645.0,122.5 638.5,126.9 640.7,119.4 634.5,114.6 642.4,114.4"/><text class="t-note" x="625" y="98">query</text><text class="t-note bad" x="346" y="66">stuck here</text><text class="t-note" x="552" y="204">true nearest</text><text class="t-note" x="400" y="142">a long link</text><text class="t-note" x="400" y="159">here fixes it</text><text class="t-note" x="36" y="180">entry</text></svg><figcaption>A local minimum. Greedy search moves E, a, b, and stops at b because its only neighbour is farther from the query. The true nearest neighbour N can only be reached by first moving away from the query, through c, d and e, which greedy search never does.</figcaption></figure>

Greedy search only ever moves closer, so it never takes the step *away* from the query that the route through c needs. The fix is not a smarter walk; it is a better graph. A single long link from b towards N removes the dead end. Long links across the space are what make a graph navigable.

Navigable Small World (NSW) graphs, introduced by Yury Malkov and colleagues in the early 2010s, get those long links for free: vectors are inserted one at a time, each linked to its nearest neighbours *among the vectors already inserted*. Early vectors are linked when the graph is sparse, so their links are long; later vectors are linked when it is dense, so theirs are short.

NSW has two weaknesses:

- **Local minima still happen.** Fewer, but they do: more friends per vertex lowers the risk but makes every hop more expensive.
- **Hubs.** The early vertices collect huge friend lists, and the walk spends much of its time scanning them. Search cost grows polylogarithmically instead of logarithmically.

## HNSW: a skip list made of graphs

HNSW (Malkov and Yashunin, 2016) fixes both by applying the skip-list idea to NSW. Instead of one graph that mixes long and short links, it builds a **stack of graphs**:

- **Layer 0** holds every vector, linked to its close neighbours: short links, precise.
- **Each layer above** holds an exponentially smaller random subset, so its links, measured between far fewer vectors, are long.
- The **entry point** is a vector on the top layer.

<figure class="fig"><svg viewBox="0 0 720 400" role="img" aria-label="HNSW as three stacked layers. The search enters at the top layer, crosses one long link, drops down, takes a shorter step, drops to layer 0 and walks two short links to the nearest neighbour."><defs><marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path class="arrow" d="M0,0L10,5L0,10z"/></marker><marker id="ah-on" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path class="arrow-on" d="M0,0L10,5L0,10z"/></marker><marker id="ah-bad" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path class="arrow-bad" d="M0,0L10,5L0,10z"/></marker></defs><polygon class="plane" points="100,38 700,38 620,120 20,120"/><text class="t-label" x="104" y="30">LAYER 2 · few vectors, long links</text><polygon class="plane" points="100,168 700,168 620,250 20,250"/><text class="t-label" x="104" y="160">LAYER 1</text><polygon class="plane" points="100,298 700,298 620,380 20,380"/><text class="t-label" x="104" y="290">LAYER 0 · every vector, short links</text><line class="drop" x1="150" y1="89" x2="150" y2="327"/><line class="drop" x1="560" y1="77" x2="560" y2="315"/><line class="drop" x1="300" y1="235" x2="300" y2="343"/><line class="drop" x1="430" y1="199" x2="430" y2="307"/><line class="edge" x1="150" y1="208" x2="300" y2="224"/><line class="edge" x1="300" y1="224" x2="430" y2="188"/><line class="edge" x1="300" y1="224" x2="560" y2="196"/><line class="edge" x1="150" y1="338" x2="178" y2="362"/><line class="edge" x1="150" y1="338" x2="215" y2="320"/><line class="edge" x1="215" y1="320" x2="262" y2="362"/><line class="edge" x1="262" y1="362" x2="300" y2="354"/><line class="edge" x1="178" y1="362" x2="262" y2="362"/><line class="edge" x1="300" y1="354" x2="370" y2="358"/><line class="edge" x1="370" y1="358" x2="395" y2="332"/><line class="edge" x1="395" y1="332" x2="430" y2="318"/><line class="edge" x1="522" y1="314" x2="560" y2="326"/><line class="edge" x1="370" y1="358" x2="482" y2="354"/><line class="edge" x1="560" y1="326" x2="592" y2="350"/><line class="edge" x1="592" y1="350" x2="482" y2="354"/><line class="edge" x1="395" y1="332" x2="482" y2="354"/><line class="path" x1="162.0" y1="77.6" x2="545.0" y2="66.4" marker-end="url(#ah-on)"/><line class="path" x1="548.0" y1="195.3" x2="445.0" y2="188.9" marker-end="url(#ah-on)"/><line class="path" x1="442.0" y1="317.5" x2="507.0" y2="314.7" marker-end="url(#ah-on)"/><line class="path" x1="513.5" y1="322.5" x2="492.6" y2="343.4" marker-end="url(#ah-on)"/><line class="path" x1="560.0" y1="78.0" x2="560.0" y2="181.0" marker-end="url(#ah-on)"/><line class="path" x1="430.0" y1="200.0" x2="430.0" y2="303.0" marker-end="url(#ah-on)"/><circle class="node entry" cx="150" cy="78" r="10"/><circle class="node on" cx="560" cy="66" r="10"/><circle class="node" cx="150" cy="208" r="10"/><circle class="node on" cx="560" cy="196" r="10"/><circle class="node" cx="300" cy="224" r="10"/><circle class="node on" cx="430" cy="188" r="10"/><circle class="node" cx="150" cy="338" r="10"/><circle class="node" cx="560" cy="326" r="10"/><circle class="node" cx="300" cy="354" r="10"/><circle class="node on" cx="430" cy="318" r="10"/><circle class="node" cx="215" cy="320" r="10"/><circle class="node" cx="262" cy="362" r="10"/><circle class="node" cx="370" cy="358" r="10"/><circle class="node hit" cx="482" cy="354" r="10"/><circle class="node on" cx="522" cy="314" r="10"/><circle class="node" cx="395" cy="332" r="10"/><circle class="node" cx="178" cy="362" r="10"/><circle class="node" cx="592" cy="350" r="10"/><polygon class="query" points="500.0,341.0 502.2,346.9 508.6,347.2 503.6,351.2 505.3,357.3 500.0,353.8 494.7,357.3 496.4,351.2 491.4,347.2 497.8,346.9"/><text class="t-note" x="36" y="394">entry point</text><circle class="node entry" cx="24" cy="389" r="6"/><polygon class="query" points="140.0,382.0 141.8,386.6 146.7,386.8 142.9,389.9 144.1,394.7 140.0,392.0 135.9,394.7 137.1,389.9 133.3,386.8 138.2,386.6"/><text class="t-note" x="150" y="394">query</text><circle class="node hit" cx="224" cy="389" r="6"/><text class="t-note" x="236" y="394">nearest neighbour found</text></svg><figcaption>HNSW search: one long hop on the top layer, drop down, one shorter hop, drop to layer 0 and finish with short steps to the nearest neighbour.</figcaption></figure>

Search walks the stack top down. On each upper layer it runs the greedy walk to a local minimum using only the closest vertex (a beam of width 1), then drops that vertex down one layer and starts again from it. On layer 0 it widens the beam to **efSearch** candidates and returns the best $$k$$.

The top layers do the long-distance travel in a few hops over very few vertices; layer 0 does the local refinement. Because the number of vertices shrinks geometrically going up, the number of hops grows slowly with $$N$$, close to logarithmically in practice on well-behaved data. Each vector also has a **capped** number of links per layer, and upper-layer links are chosen among vectors at that layer's scale, so HNSW avoids the uncontrolled high-degree hubs that plain NSW grows.

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

The single vector on layer 4 is the entry point. Five layers are enough for 200,000 vectors, and a billion would need only about seven: the top layer index grows as $$\ln N / \ln M$$.

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

HNSW has three parameters. Two are fixed when you build the index; one you choose per query. In plain words: **M** is how many roads each vector gets, **efConstruction** is how carefully those roads are planned, and **efSearch** is how many routes a search keeps open at once.

| Parameter | Set when | Controls | Costs |
|---|---|---|---|
| **M** | build | links per vector (2M on layer 0) | memory, build time, time per hop |
| **efConstruction** | build | beam width while inserting, which sets graph quality and so the recall you can reach | build and re-index time |
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

For scale: exact search over the same 200,000 vectors, one query at a time on one thread, took **1.39 ms**. HNSW at M = 16, efSearch = 64 took **0.085 ms** at 0.93 recall, 16 times faster. The gap widens with size: brute force scales linearly with $$N$$, while a well-tuned HNSW index typically shows close to logarithmic growth in practice (how close depends on the data, M, efSearch and the recall target).

### efConstruction: pay once, at build time

With M = 16 and efSearch fixed at 32:

| efConstruction | Build time | Recall@10 | Query time |
|---|---|---|---|
| 16 | 0.76 s | 0.635 | 0.042 ms |
| 32 | 1.04 s | 0.687 | 0.040 ms |
| 64 | 2.04 s | 0.776 | 0.045 ms |
| 128 | 3.90 s | 0.810 | 0.047 ms |
| 256 | 7.37 s | 0.814 | 0.046 ms |

A better-built graph gives better recall at **the same query cost**: 18 points between 16 and 128, with no change in query time. So efConstruction is not only a build-time setting: it decides how good the graph is, and therefore how much recall every later query can get. Then it plateaus. Doubling efConstruction from 128 to 256 doubled the build time for less than half a point. Around 100 to 200 is where most production defaults sit, and this curve shows why.

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
- **It wants fast random access.** A graph walk jumps to unpredictable places in memory, so HNSW is fastest when the graph and the vectors it touches are in RAM, and that is expensive at scale. Engines soften this with quantised vectors in memory, memory-mapped or on-disk storage for the full vectors, or both. Expect some latency cost when data comes from disk.

## Summary

HNSW is a skip list whose levels are proximity graphs: it keeps the skip list's coarse-to-fine layers and swaps "move right while the key is smaller" for "move to the neighbour closer to the query". Sparse upper layers with long links get the search to the right neighbourhood in a few hops; the dense bottom layer, with a wider beam, finds the precise answer. Each vector's height is a single random draw, and each vector's links are chosen for diversity rather than raw closeness.

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
