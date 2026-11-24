---
title: "NVIDIA GPUs: Every Architecture and Every Product, with Today's Prices"
description: "Ampere, Ada Lovelace, Hopper, Blackwell, Rubin: what an NVIDIA architecture is, how one design becomes GeForce, RTX PRO, data center and Jetson products, how to read the names, and the street price of each in November 2026."
date: 2026-11-23
tags: [gpu, nvidia, hardware, blackwell, buying-guide]
motif: grid
accent: "#76b900"
---

You open a shop page to buy a graphics card and see "RTX 5070 Ti". A cloud page offers an "H100" by the hour. A paper says its model was trained on "A100s". A news story says "Blackwell" is sold out and "Rubin" is next.

These are not seven different things. They are two kinds of name. **Blackwell** and **Rubin** are names of chip designs. **RTX 5070 Ti**, **H100** and **A100** are names of products made from those designs. Once you can tell the two apart, the whole NVIDIA range fits on one page.

The prices need the same care. On 23 November 2026 the cheapest RTX 5090 in a US shop costs more than **$6,000**. Its list price is $1,999. The price that matters is the one you pay, so this piece leads with the **street price** of every product. The list price sits next to it for comparison.

> [!TIP] The whole idea in one minute
> An **architecture** is one chip design with a scientist's name: Ampere (2020), Ada Lovelace (2022), Hopper (2022), Blackwell (2024), Rubin (2026). NVIDIA cuts each design to many sizes and sells them as **product families**. GeForce is for home and RTX PRO is for work. Letter-and-number cards such as H100 and B200 are for data centers, and Jetson is for robots. The first digits or the first letter of a product name tell you its architecture. For AI work, read two numbers first: the **memory on the card** (VRAM) and its **memory bandwidth**.

## One design, many products

About every two years NVIDIA makes a new chip design and names it after a scientist. It builds that design as a small chip, a middle chip and a very large chip, and each chip goes into several products.

<figure class="fig"><svg viewBox="0 0 760 330" role="img" aria-label="One architecture, Blackwell, is sold as five product families: GeForce RTX 50, RTX PRO, data center GPUs, DGX Spark and Jetson."><rect x="290" y="20" width="180" height="56" rx="8" class="box-1"/><text x="380" y="45" class="t-title" text-anchor="middle">Blackwell</text><text x="380" y="64" class="t-tick" text-anchor="middle">one chip design</text><line class="edge" x1="380" y1="76" x2="83.0" y2="150"/><rect x="14.0" y="150" width="138" height="150" rx="8" class="box"/><text x="83.0" y="176" class="t-title" text-anchor="middle">GeForce RTX 50</text><text x="83.0" y="200" class="t-tick" text-anchor="middle">games and home AI</text><text x="83.0" y="240" class="t-note" text-anchor="middle" style="font-size:12.5px">RTX 5090 to 5050</text><line class="edge" x1="380" y1="76" x2="231.5" y2="150"/><rect x="162.5" y="150" width="138" height="150" rx="8" class="box"/><text x="231.5" y="176" class="t-title" text-anchor="middle">RTX PRO</text><text x="231.5" y="200" class="t-tick" text-anchor="middle">design, film, AI</text><text x="231.5" y="240" class="t-note" text-anchor="middle" style="font-size:12.5px">RTX PRO 6000 to 2000</text><line class="edge" x1="380" y1="76" x2="380.0" y2="150"/><rect x="311.0" y="150" width="138" height="150" rx="8" class="box"/><text x="380.0" y="176" class="t-title" text-anchor="middle">Data center</text><text x="380.0" y="200" class="t-tick" text-anchor="middle">train and serve AI</text><text x="380.0" y="240" class="t-note" text-anchor="middle" style="font-size:12.5px">B200</text><text x="380.0" y="258" class="t-note" text-anchor="middle" style="font-size:12.5px">B300</text><text x="380.0" y="276" class="t-note" text-anchor="middle" style="font-size:12.5px">GB200</text><text x="380.0" y="294" class="t-note" text-anchor="middle" style="font-size:12.5px">GB300</text><line class="edge" x1="380" y1="76" x2="528.5" y2="150"/><rect x="459.5" y="150" width="138" height="150" rx="8" class="box"/><text x="528.5" y="176" class="t-title" text-anchor="middle">DGX Spark</text><text x="528.5" y="200" class="t-tick" text-anchor="middle">a desk AI computer</text><text x="528.5" y="240" class="t-note" text-anchor="middle" style="font-size:12.5px">GB10 chip</text><text x="528.5" y="258" class="t-note" text-anchor="middle" style="font-size:12.5px">64 or 128 GB</text><line class="edge" x1="380" y1="76" x2="677.0" y2="150"/><rect x="608.0" y="150" width="138" height="150" rx="8" class="box"/><text x="677.0" y="176" class="t-title" text-anchor="middle">Jetson</text><text x="677.0" y="200" class="t-tick" text-anchor="middle">robots and cameras</text><text x="677.0" y="240" class="t-note" text-anchor="middle" style="font-size:12.5px">Jetson AGX Thor</text></svg><figcaption>One architecture, many products. The same Blackwell design is cut to different sizes and sold to gamers, professionals, data centers and robot builders.</figcaption></figure>

Two things come with a new architecture. The first is a new **manufacturing process**: smaller transistors, so more of them fit on a chip and each uses less power. The second is new **abilities**, such as special cores for AI maths. An old card cannot get those abilities through a software update, because they are built into the silicon.

## The timeline, from 2006 to 2026

<figure class="fig"><svg viewBox="0 0 760 454" role="img" aria-label="A table of NVIDIA GPU architectures from Tesla in 2006 to Rubin in 2026, with the first year, well-known products and what was new in each."><text x="20" y="24" class="t-label" text-anchor="start">ARCHITECTURE</text><text x="190" y="24" class="t-label" text-anchor="start">FIRST YEAR</text><text x="300" y="24" class="t-label" text-anchor="start">WELL-KNOWN PRODUCTS</text><text x="530" y="24" class="t-label" text-anchor="start">WHAT WAS NEW</text><rect x="14" y="40" width="732" height="28" rx="6" class="box"/><text x="24" y="59" class="t-title" text-anchor="start">Tesla</text><text x="190" y="59" class="t-val" text-anchor="start">2006</text><text x="300" y="59" class="t-note" text-anchor="start">GeForce 8</text><text x="530" y="59" class="t-tick" text-anchor="start">CUDA: GPUs run general code</text><rect x="14" y="74" width="732" height="28" rx="6" class="box"/><text x="24" y="93" class="t-title" text-anchor="start">Fermi</text><text x="190" y="93" class="t-val" text-anchor="start">2010</text><text x="300" y="93" class="t-note" text-anchor="start">GTX 400</text><text x="530" y="93" class="t-tick" text-anchor="start">Memory that corrects errors</text><rect x="14" y="108" width="732" height="28" rx="6" class="box"/><text x="24" y="127" class="t-title" text-anchor="start">Kepler</text><text x="190" y="127" class="t-val" text-anchor="start">2012</text><text x="300" y="127" class="t-note" text-anchor="start">GTX 600, K80</text><text x="530" y="127" class="t-tick" text-anchor="start">Lower power</text><rect x="14" y="142" width="732" height="28" rx="6" class="box"/><text x="24" y="161" class="t-title" text-anchor="start">Maxwell</text><text x="190" y="161" class="t-val" text-anchor="start">2014</text><text x="300" y="161" class="t-note" text-anchor="start">GTX 900</text><text x="530" y="161" class="t-tick" text-anchor="start">Much lower power</text><rect x="14" y="176" width="732" height="28" rx="6" class="box"/><text x="24" y="195" class="t-title" text-anchor="start">Pascal</text><text x="190" y="195" class="t-val" text-anchor="start">2016</text><text x="300" y="195" class="t-note" text-anchor="start">GTX 10, P100</text><text x="530" y="195" class="t-tick" text-anchor="start">Big jump for deep learning</text><rect x="14" y="210" width="732" height="28" rx="6" class="box"/><text x="24" y="229" class="t-title" text-anchor="start">Volta</text><text x="190" y="229" class="t-val" text-anchor="start">2017</text><text x="300" y="229" class="t-note" text-anchor="start">V100</text><text x="530" y="229" class="t-tick" text-anchor="start">Tensor Cores (AI maths)</text><rect x="14" y="244" width="732" height="28" rx="6" class="box"/><text x="24" y="263" class="t-title" text-anchor="start">Turing</text><text x="190" y="263" class="t-val" text-anchor="start">2018</text><text x="300" y="263" class="t-note" text-anchor="start">RTX 20, T4</text><text x="530" y="263" class="t-tick" text-anchor="start">RT Cores (ray tracing), DLSS</text><rect x="14" y="278" width="732" height="28" rx="6" class="box-1"/><text x="24" y="297" class="t-title" text-anchor="start">Ampere</text><text x="190" y="297" class="t-val" text-anchor="start">2020</text><text x="300" y="297" class="t-note" text-anchor="start">RTX 30, A100</text><text x="530" y="297" class="t-tick" text-anchor="start">Faster Tensor Cores, sparsity</text><rect x="14" y="312" width="732" height="28" rx="6" class="box-1"/><text x="24" y="331" class="t-title" text-anchor="start">Ada Lovelace</text><text x="190" y="331" class="t-val" text-anchor="start">2022</text><text x="300" y="331" class="t-note" text-anchor="start">RTX 40, L4</text><text x="530" y="331" class="t-tick" text-anchor="start">Frame generation, AV1</text><rect x="14" y="346" width="732" height="28" rx="6" class="box-2"/><text x="24" y="365" class="t-title" text-anchor="start">Hopper</text><text x="190" y="365" class="t-val" text-anchor="start">2022</text><text x="300" y="365" class="t-note" text-anchor="start">H100, H200</text><text x="530" y="365" class="t-tick" text-anchor="start">Data center only, 8-bit floats</text><rect x="14" y="380" width="732" height="28" rx="6" class="box-1"/><text x="24" y="399" class="t-title" text-anchor="start">Blackwell</text><text x="190" y="399" class="t-val" text-anchor="start">2024</text><text x="300" y="399" class="t-note" text-anchor="start">RTX 50, B200</text><text x="530" y="399" class="t-tick" text-anchor="start">4-bit floats, more memory</text><rect x="14" y="414" width="732" height="28" rx="6" class="box-2"/><text x="24" y="433" class="t-title" text-anchor="start">Rubin</text><text x="190" y="433" class="t-val" text-anchor="start">2026</text><text x="300" y="433" class="t-note" text-anchor="start">Vera Rubin</text><text x="530" y="433" class="t-tick" text-anchor="start">HBM4 memory, Vera CPU</text></svg><figcaption>Every NVIDIA architecture since 2006. Blue rows are the three you meet most in shops and clouds today. Orange rows are made for data centers only.</figcaption></figure>

You do not need all twelve rows. Four moments explain today's range.

**2006, Tesla: the GPU learns general code.** With this architecture NVIDIA released **CUDA**, a way to write normal programs for a GPU, not only pictures. Every later AI library sits on CUDA. This is the main reason NVIDIA hardware is the default choice for AI today.

**2017, Volta: cores made for AI.** Volta added **Tensor Cores**. A Tensor Core multiplies small tables of numbers in one step, which is the operation a neural network repeats billions of times. The Volta card V100 became the standard GPU of AI research for three years.

> [!DEFINITION] Tensor Core
> A part of the chip that multiplies matrices (tables of numbers) directly, much faster than the general cores. Each architecture since Volta makes them faster and teaches them smaller number formats: 16-bit, then 8-bit in Hopper, then 4-bit in Blackwell. Smaller numbers mean more work per second and per watt.

**2018, Turing: the name changes from GTX to RTX.** Turing added **RT Cores** for ray tracing, a way to draw light that follows real rays. It also brought DLSS, where a neural network on the Tensor Cores makes a low-resolution picture sharp. Cards with these cores are called **RTX**. The older name **GTX** means a card without them.

**2022, the road splits.** Until then, one architecture served both games and data centers. In 2022 NVIDIA released two: **Ada Lovelace** for games, workstations and small servers, and **Hopper** for the largest AI servers only. No Hopper game card exists. Blackwell in 2024 joined the two roads again, and Rubin in 2026 has so far shipped only for data centers.

## The three architectures you meet most

### Ampere (2020)

Ampere is the oldest architecture still in wide use. Its home products are the **GeForce RTX 30** series. Its workstation cards are the **RTX A** series, where the RTX A6000 has 48 GB. Its data center cards are the **A100** and **A10**. Many papers from 2020 to 2023 report results on the A100, which has 40 GB or 80 GB of memory.

Ampere is also inside the **Jetson Orin** boards for robots, which are still sold new. So an architecture from 2020 can be in a product you buy today.

### Ada Lovelace (2022) and Hopper (2022)

Ada Lovelace gave the **GeForce RTX 40** series, the **RTX Ada** workstation cards and two popular server cards. The **L4** is small, uses 72 watts and has 24 GB. The **L40S** has 48 GB and is a common choice for serving models. Ada added frame generation, where the card invents whole frames between the real ones.

<figure class="fig photo"><img src="/img/writings/gpu/rtx3090v4090.jpg" width="960" height="797" alt="Two large graphics cards with three fans each, an RTX 3090 above and a wider RTX 4090 below." loading="lazy" /><figcaption>An RTX 3090 (top, Ampere) and an RTX 4090 (bottom, Ada Lovelace). Both have 24 GB. The top tier grew with each generation. <span class="credit">Photo: <a href="https://commons.wikimedia.org/wiki/File:Size_comparison_between_RTX_3090_and_RTX_4090.jpg">Benlisquare</a>, CC BY-SA 4.0, Wikimedia Commons.</span></figcaption></figure>

Hopper gave the **H100** (80 GB) and, in 2024, the **H200** (141 GB). The H100 is the card behind the large language models of 2023 and 2024. Hopper's Tensor Cores can compute with 8-bit numbers, which roughly doubles the speed of some training work against 16-bit.

### Blackwell (2024)

Blackwell is the current architecture in shops. It covers every family in the figure above. They are **GeForce RTX 50**, **RTX PRO**, the data center cards **B200** and **B300**, the small **DGX Spark** computer and the **Jetson AGX Thor** robot board.

Its data center cards are much larger than before. The B200 has about 180 GB of memory and uses about 1,000 watts. Blackwell's Tensor Cores add 4-bit numbers. A faster version, **Blackwell Ultra** (the B300, with 288 GB), followed in 2025.

### Rubin (2026), and what comes after

Rubin is the newest architecture. One Rubin GPU has 288 GB of a newer, faster memory type called HBM4, explained below. It comes with a new NVIDIA processor named **Vera**. Reports in November 2026 agree that Rubin systems have started to ship to a few clouds, and disagree on how fast the volume will grow. No Rubin card for games has been announced. NVIDIA's public roadmap names **Rubin Ultra** next, then an architecture called **Feynman**.

## How to read a product name

### GeForce: the number tells you two things

<figure class="fig"><svg viewBox="0 0 760 300" role="img" aria-label="How to read the name GeForce RTX 5070 Ti: GeForce is the family, RTX means ray tracing and AI cores, 50 is the generation, 70 is the tier and Ti is a small step up."><rect x="60" y="30" width="110" height="54" rx="8" class="box-1"/><text x="115" y="64" class="t-title" text-anchor="middle" style="font-size:20px">GeForce</text><text x="115" y="108" class="t-tick" text-anchor="middle">the family:</text><text x="115" y="125" class="t-tick" text-anchor="middle">for games</text><rect x="215" y="30" width="110" height="54" rx="8" class="box-1"/><text x="270" y="64" class="t-title" text-anchor="middle" style="font-size:20px">RTX</text><text x="270" y="108" class="t-tick" text-anchor="middle">has ray tracing</text><text x="270" y="125" class="t-tick" text-anchor="middle">and AI cores</text><rect x="340" y="30" width="110" height="54" rx="8" class="box-1"/><text x="395" y="64" class="t-title" text-anchor="middle" style="font-size:20px">50</text><text x="395" y="108" class="t-tick" text-anchor="middle">the generation:</text><text x="395" y="125" class="t-tick" text-anchor="middle">Blackwell</text><rect x="455" y="30" width="110" height="54" rx="8" class="box-1"/><text x="510" y="64" class="t-title" text-anchor="middle" style="font-size:20px">70</text><text x="510" y="108" class="t-tick" text-anchor="middle">the tier:</text><text x="510" y="125" class="t-tick" text-anchor="middle">higher is faster</text><rect x="575" y="30" width="110" height="54" rx="8" class="box-1"/><text x="630" y="64" class="t-title" text-anchor="middle" style="font-size:20px">Ti</text><text x="630" y="108" class="t-tick" text-anchor="middle">a small</text><text x="630" y="125" class="t-tick" text-anchor="middle">step up</text><line class="axis" x1="20" y1="158" x2="740" y2="158"/><text x="20" y="186" class="t-label" text-anchor="start">Generation number</text><text x="20" y="268" class="t-label" text-anchor="start">Tier number</text><rect x="200" y="168" width="128" height="46" rx="8" class="box"/><text x="264" y="187" class="t-title" text-anchor="middle">RTX 20</text><text x="264" y="205" class="t-tick" text-anchor="middle">Turing</text><rect x="338" y="168" width="128" height="46" rx="8" class="box"/><text x="402" y="187" class="t-title" text-anchor="middle">RTX 30</text><text x="402" y="205" class="t-tick" text-anchor="middle">Ampere</text><rect x="476" y="168" width="128" height="46" rx="8" class="box"/><text x="540" y="187" class="t-title" text-anchor="middle">RTX 40</text><text x="540" y="205" class="t-tick" text-anchor="middle">Ada Lovelace</text><rect x="614" y="168" width="128" height="46" rx="8" class="box"/><text x="678" y="187" class="t-title" text-anchor="middle">RTX 50</text><text x="678" y="205" class="t-tick" text-anchor="middle">Blackwell</text><rect x="200" y="232" width="100" height="46" rx="8" class="box"/><text x="250" y="251" class="t-title" text-anchor="middle">50</text><text x="250" y="269" class="t-tick" text-anchor="middle">entry</text><rect x="310" y="232" width="100" height="46" rx="8" class="box"/><text x="360" y="251" class="t-title" text-anchor="middle">60</text><text x="360" y="269" class="t-tick" text-anchor="middle">mainstream</text><rect x="420" y="232" width="100" height="46" rx="8" class="box"/><text x="470" y="251" class="t-title" text-anchor="middle">70</text><text x="470" y="269" class="t-tick" text-anchor="middle">upper middle</text><rect x="530" y="232" width="100" height="46" rx="8" class="box"/><text x="580" y="251" class="t-title" text-anchor="middle">80</text><text x="580" y="269" class="t-tick" text-anchor="middle">high end</text><rect x="640" y="232" width="100" height="46" rx="8" class="box-3"/><text x="690" y="251" class="t-title" text-anchor="middle">90</text><text x="690" y="269" class="t-tick" text-anchor="middle">the largest</text></svg><figcaption>How to read a card name. The first digits give the generation, and so the architecture. The last two digits give the tier inside that generation.</figcaption></figure>

A GeForce name has a generation and a tier. In **RTX 5070**, "50" is the generation, which is Blackwell, and "70" is the tier. A higher tier in the same generation is always faster and almost always has more memory.

Three suffixes appear after the number. **Ti** is a faster version of the same tier. **Super** is a refresh sold about a year later. **Laptop GPU** marks a version for notebooks.

> [!WARNING] A laptop card is not the desktop card
> The laptop RTX 5090 has 24 GB of memory and uses a smaller chip than the desktop RTX 5090, which has 32 GB. The names match, the hardware does not. Compare laptop cards only with other laptop cards.

The tier only compares cards inside one generation. An RTX 4090 from 2022 is still faster than an RTX 5070 from 2025. The largest chip of an old generation beats the middle chip of a new one.

### Data center: the letter tells you the architecture

Data center cards have a letter and a number. The letter is the first letter of the architecture, and a larger number is a larger card.

| Name | Architecture | Memory | What people use it for |
|---|---|---|---|
| T4 | Turing | 16 GB | Cheap serving of small models, video |
| A10, A100 | Ampere | 24 GB, 40 or 80 GB | Older training and serving |
| L4, L40S | Ada **L**ovelace | 24 GB, 48 GB | Serving, image models, fine-tuning |
| H100, H200 | Hopper | 80 GB, 141 GB | Training and serving large models |
| B200, B300 | Blackwell | about 180 GB, 288 GB | The largest models of 2025 and 2026 |

A second prefix, **G**, means the GPU is joined to an NVIDIA processor named Grace on one board. So a **GB200** is Grace plus Blackwell GPUs, and a **GB300** is Grace plus Blackwell Ultra GPUs. The name **NVL72** after it means a full rack with 72 GPUs wired together to work as one machine.

### Workstation: the architecture is written in the name

Workstation cards were called **Quadro** until 2020. Since then the architecture appears in the name itself. **RTX A6000** is Ampere, **RTX 6000 Ada** is Ada Lovelace and **RTX PRO 6000 Blackwell** is Blackwell. These cards use the same chips as GeForce cards, with much more memory and certified drivers. Their shape lets four of them sit side by side.

## The product map

<figure class="fig"><svg viewBox="0 0 760 318" role="img" aria-label="A map of NVIDIA products by where they live: laptop, home PC, workstation, desk box, server, full rack, and robot or car."><text x="24" y="24" class="t-label" text-anchor="start">WHERE IT LIVES</text><text x="250" y="24" class="t-label" text-anchor="start">PRODUCTS</text><text x="570" y="24" class="t-label" text-anchor="start">MEMORY FOR THE GPU</text><rect x="14" y="36" width="732" height="33" rx="6" class="box"/><text x="24" y="57" class="t-title" text-anchor="start">In a laptop</text><text x="250" y="57" class="t-note" text-anchor="start">GeForce RTX 50 Laptop</text><text x="570" y="57" class="t-val" text-anchor="start">up to 24 GB</text><rect x="14" y="76" width="732" height="33" rx="6" class="box-1"/><text x="24" y="97" class="t-title" text-anchor="start">In a home PC</text><text x="250" y="97" class="t-note" text-anchor="start">GeForce RTX 5050 to 5090</text><text x="570" y="97" class="t-val" text-anchor="start">8 to 32 GB</text><rect x="14" y="116" width="732" height="33" rx="6" class="box-1"/><text x="24" y="137" class="t-title" text-anchor="start">In a workstation</text><text x="250" y="137" class="t-note" text-anchor="start">RTX PRO 2000 to 6000</text><text x="570" y="137" class="t-val" text-anchor="start">16 to 96 GB</text><rect x="14" y="156" width="732" height="33" rx="6" class="box-3"/><text x="24" y="177" class="t-title" text-anchor="start">On a desk, as a box</text><text x="250" y="177" class="t-note" text-anchor="start">DGX Spark, DGX Station</text><text x="570" y="177" class="t-val" text-anchor="start">64 GB shared and up</text><rect x="14" y="196" width="732" height="33" rx="6" class="box-2"/><text x="24" y="217" class="t-title" text-anchor="start">In a server</text><text x="250" y="217" class="t-note" text-anchor="start">L4, L40S, H100, H200, B200, B300</text><text x="570" y="217" class="t-val" text-anchor="start">24 to 288 GB each</text><rect x="14" y="236" width="732" height="33" rx="6" class="box-2"/><text x="24" y="257" class="t-title" text-anchor="start">In a full rack</text><text x="250" y="257" class="t-note" text-anchor="start">GB300 NVL72, Vera Rubin NVL72</text><text x="570" y="257" class="t-val" text-anchor="start">72 GPUs together</text><rect x="14" y="276" width="732" height="33" rx="6" class="box-3"/><text x="24" y="297" class="t-title" text-anchor="start">In a robot or car</text><text x="250" y="297" class="t-note" text-anchor="start">Jetson Orin, Jetson Thor, DRIVE</text><text x="570" y="297" class="t-val" text-anchor="start">8 to 128 GB shared</text></svg><figcaption>The whole product range by where the GPU lives. Memory grows as you go down from a laptop to a server.</figcaption></figure>

**GeForce** is for home computers. Most people who play games or run a small model at home buy one of these. The current range goes from the RTX 5050 (8 GB) to the RTX 5090 (32 GB).

<figure class="fig photo"><img src="/img/writings/gpu/rtx5060ti.jpg" width="1280" height="853" alt="A GeForce RTX 5060 Ti with 16 GB, made by PNY: two fans, a metal bracket with the video ports, and the gold edge connector that goes into the motherboard." loading="lazy" /><figcaption>A home card: the GeForce RTX 5060 Ti with 16 GB. NVIDIA makes the chip, and board makers such as PNY build the card around it. <span class="credit">Photo: <a href="https://commons.wikimedia.org/wiki/File:Nvidia_GeForce_RTX_5060_Ti_16GB,_PNY_Overclocked_Dual_Fan,_front.jpg">FreeMediaKid!</a>, CC BY-SA 4.0, Wikimedia Commons.</span></figcaption></figure>

**RTX PRO** is for people who work on one machine: engineers, film studios, researchers. The RTX PRO 6000 Blackwell has 96 GB, three times the memory of the RTX 5090, on a similar chip.

<figure class="fig photo"><img src="/img/writings/gpu/ada6000.jpg" width="1280" height="720" alt="Four slim RTX 6000 Ada workstation cards side by side in one open computer." loading="lazy" /><figcaption>Four RTX 6000 Ada workstation cards in one machine. Each has 48 GB and a slim shape with one fan at the end, so they fit side by side. <span class="credit">Photo: <a href="https://commons.wikimedia.org/wiki/File:4%C3%97_NVIDIA_RTX%E2%84%A2_6000_der_Ada-Generation_(%E6%9E%81%E5%AE%A2%E6%B9%BEGeekerwan)_01.png">Geekerwan</a>, CC BY 3.0, Wikimedia Commons.</span></figcaption></figure>

**DGX Spark** is a small box, about the size of a book, with a Blackwell GPU and a processor on one chip. They share 128 GB of memory. It is slower than a large card, but it can load models that no home card can hold. **DGX Station** is its larger brother for a desk.

<figure class="fig photo"><img src="/img/writings/gpu/spark.jpg" width="1280" height="790" alt="A small gold-coloured metal box on a wooden desk: the NVIDIA DGX Spark." loading="lazy" /><figcaption>The DGX Spark: a whole AI computer in a box that fits in one hand. <span class="credit">Photo: <a href="https://commons.wikimedia.org/wiki/File:Nvidia_DGX_Spark_oblique_view_dllu.jpg">Daniel Lu (dllu)</a>, CC BY-SA 4.0, Wikimedia Commons.</span></figcaption></figure>

**Data center cards** sit in servers, usually eight to a server. You cannot put one in a home PC: they have no fan and need a server's airflow or liquid cooling. Most people never buy one. They rent one by the hour from a cloud.

<figure class="fig photo"><img src="/img/writings/gpu/h100.jpg" width="1280" height="720" alt="Four gold-coloured H100 cards standing in a row. They have no fans." loading="lazy" /><figcaption>H100 cards. They have no fans: the server pushes air through them. <span class="credit">Photo: <a href="https://commons.wikimedia.org/wiki/File:NVIDIA_H100_(%E6%9E%81%E5%AE%A2%E6%B9%BEGeekerwan)_011.png">Geekerwan</a>, CC BY 3.0, Wikimedia Commons.</span></figcaption></figure>

<figure class="fig photo tall"><img src="/img/writings/gpu/dgxb200.jpg" width="960" height="974" alt="A large server board with eight tall black GPU modules, labelled HGX B200." loading="lazy" /><figcaption>Eight B200 GPUs on one board, the inside of a B200 server. <span class="credit">Photo: <a href="https://commons.wikimedia.org/wiki/File:Nvidia_DGX-B200-HGX.jpg">Pokiiri</a>, CC BY-SA 4.0, Wikimedia Commons.</span></figcaption></figure>

**Racks** such as the GB300 NVL72 are sold whole. One rack holds 72 GPUs and 36 processors, is cooled by liquid and draws about as much power as a hundred homes. Companies that train the largest models buy thousands of them.

<figure class="fig photo tall"><img src="/img/writings/gpu/gb200.jpg" width="640" height="1100" alt="A tall black server rack filled with flat trays: an NVIDIA GB200 rack." loading="lazy" /><figcaption>A GB200 rack with 72 Blackwell GPUs and 36 Grace processors, cooled by liquid. It is sold as one machine. <span class="credit">Photo: <a href="https://commons.wikimedia.org/wiki/File:Nvidia_DGX_GB200.jpg">Pokiiri</a>, CC BY-SA 4.0, Wikimedia Commons.</span></figcaption></figure>

**Jetson** is a small board with a GPU, a processor and memory, for robots, drones and cameras. **DRIVE** is the same idea for cars.

<figure class="fig photo"><img src="/img/writings/gpu/jetson.jpg" width="1280" height="968" alt="A small grey box with cooling fins next to its retail carton: the Jetson AGX Orin developer kit." loading="lazy" /><figcaption>The Jetson AGX Orin developer kit, a complete computer for a robot. <span class="credit">Photo: <a href="https://commons.wikimedia.org/wiki/File:NVidia_Jetson_Orin_AGX.png">Auledas</a>, CC BY 4.0, Wikimedia Commons.</span></figcaption></figure>

## Four words on every spec sheet

A spec sheet for a GPU uses four terms again and again: SM, VRAM, HBM and memory bandwidth. Together they tell you how much work a card does and how large a model it holds.

### SM: the unit a GPU is built from

A GPU is not one large processor. It is one small processing block copied many times. NVIDIA calls that block a **streaming multiprocessor**, or SM.

> [!DEFINITION] SM (streaming multiprocessor)
> One block of a GPU chip. It holds a group of general cores (the "CUDA cores"), a few Tensor Cores for matrix maths, its own small fast memory and a scheduler. A bigger chip of the same architecture is the same SM, copied more times.

| GPU | Architecture | SMs | CUDA cores in one SM | CUDA cores in total |
|---|---|---|---|---|
| A100 | Ampere | 108 | 64 | 6,912 |
| RTX 4090 | Ada Lovelace | 128 | 128 | 16,384 |
| H100 | Hopper | 132 | 128 | 16,896 |
| RTX 5090 | Blackwell | 170 | 128 | 21,760 |

This table explains the product range. Inside one architecture, a higher tier is mostly the same SM copied more times, with more memory beside it. A new architecture changes what one SM can do. One SM of the RTX 5090 has 128 CUDA cores, 4 Tensor Cores and 1 RT Core.

Do not compare core counts between architectures. The A100 has fewer than half the CUDA cores of the RTX 4090. It still trains larger models, because it has 80 GB of memory against 24 GB. The site's [GPU Programming](/books/gpu-programming) book shows how code runs on an SM.

### VRAM and RAM

A computer with a graphics card has two separate memories.

| | RAM | VRAM |
|---|---|---|
| Where it sits | On the motherboard, next to the CPU | On the graphics card, next to the GPU chip |
| Who uses it | The CPU and every program | Only the GPU |
| Typical size at home | 16 to 64 GB | 8 to 32 GB |
| Can you add more later | Yes | No, it is soldered to the card |

> [!DEFINITION] VRAM
> The memory on the graphics card itself. The GPU can only compute on data that is in VRAM. To use a model, the computer copies its weights from the disk into VRAM once, and they stay there.

So 64 GB of RAM does not help a card with 8 GB of VRAM. If a model does not fit in VRAM, software can keep part of it in RAM and run that part on the CPU. It works, and it is many times slower.

Two products break this rule. The **DGX Spark** and the **Jetson** boards have one pool of memory that the CPU and the GPU share. That is why a DGX Spark can give 128 GB to a model when no home card has more than 32 GB.

<figure class="fig"><svg viewBox="0 0 760 344" role="img" aria-label="A bar chart of the memory on one GPU, from 8 gigabytes on an RTX 5060 to 288 gigabytes on a B300."><text x="14" y="22" class="t-title" text-anchor="start">Memory on one GPU, gigabytes (log scale)</text><text x="240" y="54" class="t-note" text-anchor="end">RTX 5060 (8 GB)</text><rect x="250" y="41" width="48.7" height="18" rx="3" class="s1"/><text x="306.6578474872212" y="54" class="t-val" text-anchor="start">8 GB</text><text x="240" y="84" class="t-note" text-anchor="end">RTX 5070 (12 GB)</text><rect x="250" y="71" width="87.3" height="18" rx="3" class="s1"/><text x="345.2797538165611" y="84" class="t-val" text-anchor="start">12 GB</text><text x="240" y="114" class="t-note" text-anchor="end">RTX 5080 (16 GB)</text><rect x="250" y="101" width="114.7" height="18" rx="3" class="s1"/><text x="372.6824324507459" y="114" class="t-val" text-anchor="start">16 GB</text><text x="240" y="144" class="t-note" text-anchor="end">L4 (24 GB)</text><rect x="250" y="131" width="153.3" height="18" rx="3" class="s2"/><text x="411.3043387800858" y="144" class="t-val" text-anchor="start">24 GB</text><text x="240" y="174" class="t-note" text-anchor="end">RTX 5090 (32 GB)</text><rect x="250" y="161" width="180.7" height="18" rx="3" class="s1"/><text x="438.7070174142707" y="174" class="t-val" text-anchor="start">32 GB</text><text x="240" y="204" class="t-note" text-anchor="end">L40S (48 GB)</text><rect x="250" y="191" width="219.3" height="18" rx="3" class="s2"/><text x="477.3289237436106" y="204" class="t-val" text-anchor="start">48 GB</text><text x="240" y="234" class="t-note" text-anchor="end">H100 (80 GB)</text><rect x="250" y="221" width="268.0" height="18" rx="3" class="s2"/><text x="525.9867712308317" y="234" class="t-val" text-anchor="start">80 GB</text><text x="240" y="264" class="t-note" text-anchor="end">RTX PRO 6000 (96 GB)</text><rect x="250" y="251" width="285.4" height="18" rx="3" class="s2"/><text x="543.3535087071353" y="264" class="t-val" text-anchor="start">96 GB</text><text x="240" y="294" class="t-note" text-anchor="end">H200 (141 GB)</text><rect x="250" y="281" width="322.0" height="18" rx="3" class="s2"/><text x="579.9700074645491" y="294" class="t-val" text-anchor="start">141 GB</text><text x="240" y="324" class="t-note" text-anchor="end">B300 (288 GB)</text><rect x="250" y="311" width="390.0" height="18" rx="3" class="s2"/><text x="648.0" y="324" class="t-val" text-anchor="start">288 GB</text></svg><figcaption>Memory on one GPU. For AI work this number matters more than any other, because a model must fit in it.</figcaption></figure>

### GDDR and HBM: two kinds of VRAM

Home and workstation cards use **GDDR** memory: flat chips soldered around the GPU chip. The RTX 50 cards use the newest kind, GDDR7.

<figure class="fig photo"><img src="/img/writings/gpu/pcb5090.jpg" width="1280" height="538" alt="The bare circuit boards of the RTX 5080 and RTX 5090 Founders Edition cards. Each has a large square chip in the middle with small black memory chips around it." loading="lazy" /><figcaption>The boards of the RTX 5080 and RTX 5090 without their coolers. The large square is the GPU chip. The small black rectangles around it are the GDDR7 memory chips: that is the VRAM. <span class="credit">Photo: <a href="https://commons.wikimedia.org/wiki/File:Nvidia_RTX_5080_5090_FE_PCB.png">Geekerwan</a>, CC BY 3.0, Wikimedia Commons.</span></figcaption></figure>

> [!DEFINITION] HBM (high-bandwidth memory)
> Memory chips stacked on top of each other, with thousands of vertical wires through the stack, placed right beside the GPU chip. The many short wires let far more data move each second than GDDR can. Data center GPUs use HBM.

Each generation of HBM is faster. The A100 uses HBM2e, the H100 uses HBM3, the H200 and B200 use HBM3e and Rubin uses HBM4. HBM is hard to make, and the whole AI industry wants it. It is one reason a data center card costs so much more than a home card.

### Memory bandwidth

> [!DEFINITION] Memory bandwidth
> How many gigabytes the GPU chip can read from its VRAM in one second, written GB/s. VRAM size says how much data the card holds. Bandwidth says how fast the chip can reach that data.

Think of VRAM as a warehouse and of bandwidth as the width of its door. A large warehouse with a narrow door holds a lot and serves it slowly.

<figure class="fig"><svg viewBox="0 0 760 392" role="img" aria-label="A bar chart of memory bandwidth, from 273 gigabytes per second on a DGX Spark to about 7,700 on a B200."><text x="14" y="22" class="t-title" text-anchor="start">Memory bandwidth of one GPU, gigabytes per second (log scale)</text><text x="240" y="54" class="t-note" text-anchor="end">DGX Spark (LPDDR5X)</text><rect x="250" y="41" width="51.7" height="18" rx="3" class="s3"/><text x="309.7415448621524" y="54" class="t-val" text-anchor="start">273 GB/s</text><text x="240" y="84" class="t-note" text-anchor="end">RTX 5060 Ti (GDDR7)</text><rect x="250" y="71" width="101.9" height="18" rx="3" class="s1"/><text x="359.91267014102766" y="84" class="t-val" text-anchor="start">448 GB/s</text><text x="240" y="114" class="t-note" text-anchor="end">RTX 5070 (GDDR7)</text><rect x="250" y="101" width="143.0" height="18" rx="3" class="s1"/><text x="400.9822447769835" y="114" class="t-val" text-anchor="start">672 GB/s</text><text x="240" y="144" class="t-note" text-anchor="end">L40S (GDDR6)</text><rect x="250" y="131" width="168.4" height="18" rx="3" class="s2"/><text x="426.4378917931549" y="144" class="t-val" text-anchor="start">864 GB/s</text><text x="240" y="174" class="t-note" text-anchor="end">RTX 5070 Ti (GDDR7)</text><rect x="250" y="161" width="172.1" height="18" rx="3" class="s1"/><text x="430.1215717121954" y="174" class="t-val" text-anchor="start">896 GB/s</text><text x="240" y="204" class="t-note" text-anchor="end">RTX 5080 (GDDR7)</text><rect x="250" y="191" width="179.1" height="18" rx="3" class="s1"/><text x="437.10986201935145" y="204" class="t-val" text-anchor="start">960 GB/s</text><text x="240" y="234" class="t-note" text-anchor="end">RTX 5090 (GDDR7)</text><rect x="250" y="221" width="242.3" height="18" rx="3" class="s1"/><text x="500.3304732833632" y="234" class="t-val" text-anchor="start">1,792 GB/s</text><text x="240" y="264" class="t-note" text-anchor="end">A100 80 GB (HBM2e)</text><rect x="250" y="251" width="253.5" height="18" rx="3" class="s2"/><text x="511.4536248084442" y="264" class="t-val" text-anchor="start">2,000 GB/s</text><text x="240" y="294" class="t-note" text-anchor="end">H100 (HBM3)</text><rect x="250" y="281" width="305.7" height="18" rx="3" class="s2"/><text x="563.7003579204395" y="294" class="t-val" text-anchor="start">3,350 GB/s</text><text x="240" y="324" class="t-note" text-anchor="end">H200 (HBM3e)</text><rect x="250" y="311" width="342.1" height="18" rx="3" class="s2"/><text x="600.1298830886274" y="324" class="t-val" text-anchor="start">4,800 GB/s</text><text x="240" y="354" class="t-note" text-anchor="end">B200 (HBM3e)</text><rect x="250" y="341" width="390.0" height="18" rx="3" class="s2"/><text x="648.0" y="354" class="t-val" text-anchor="start">7,700 GB/s</text><text x="14" y="382" class="t-tick" text-anchor="start">Blue: GeForce. Orange: workstation and data center. Green: small computers.</text></svg><figcaption>Memory bandwidth, with the memory type in brackets. Every card with HBM is above every card with GDDR. Data center values are for the server versions; the B200 is given as 7,700 or 8,000 GB/s, depending on the source.</figcaption></figure>

The DGX Spark is the clearest case. It holds 128 GB, four times an RTX 5090, and its bandwidth is 273 GB/s, less than a sixth of the RTX 5090. It can load very large models and it runs them slowly.

## What this means for a language model

Two questions decide if a card is good for a language model. Does the model fit, and how fast does it write?

### Does it fit: VRAM

The weights of a model take a fixed space. At 16-bit precision, one billion weights need 2 GB. Most people at home run models compressed to 4 bits, which need about 0.6 GB for one billion weights. Keep about a fifth of the VRAM free for the conversation itself. The site's piece on the [KV cache](/writings/llm-inference-2-kv-cache) explains that part.

With those two rules, the VRAM of a card gives the largest model it runs.

| VRAM | Cards | Largest model at 4 bits | Largest model at 16 bits |
|---|---|---|---|
| 8 GB | RTX 5050, RTX 5060 | about 10 billion weights | about 3 billion |
| 12 GB | RTX 5070 | about 16 billion | about 5 billion |
| 16 GB | RTX 5060 Ti, RTX 5070 Ti, RTX 5080 | about 21 billion | about 6 billion |
| 24 GB | RTX 3090, RTX 4090, L4 | about 32 billion | about 10 billion |
| 32 GB | RTX 5090 | about 43 billion | about 13 billion |
| 48 GB | L40S, RTX PRO 5000 | about 64 billion | about 19 billion |
| 80 GB | H100 | about 107 billion | about 32 billion |
| 96 GB | RTX PRO 6000 Blackwell | about 128 billion | about 38 billion |

This is why the RTX 5060 Ti with 16 GB is a common first card for AI at home. It is also why an old RTX 3090 with 24 GB still sells for more than a new RTX 5070 Ti.

### How fast it writes: bandwidth

A language model writes one token (a word piece) at a time. For each token, the GPU reads every weight of the model once. So the bandwidth sets a ceiling on speed:

$$
\text{tokens per second} \le \frac{\text{memory bandwidth}}{\text{size of the weights}}
$$

Take a model with 8 billion weights at 4 bits, which is about 5 GB.

| Card | Bandwidth | Ceiling for the 5 GB model |
|---|---|---|
| DGX Spark | 273 GB/s | about 55 tokens per second |
| RTX 5060 Ti | 448 GB/s | about 90 |
| RTX 5070 | 672 GB/s | about 134 |
| RTX 5080 | 960 GB/s | about 192 |
| RTX 5090 | 1,792 GB/s | about 358 |
| H100 | 3,350 GB/s | about 670 |

These are ceilings from the formula, not measurements. Real speed is lower, because the chip also has to compute. The order of the cards is what the table teaches. It also shows why the RTX 5080 and the RTX 5060 Ti are not equal for AI. Both have 16 GB and hold the same models, and the RTX 5080 has about twice the bandwidth.

A server works in a different way. It answers many people in one pass over the weights, so one read of the weights produces many tokens. That is why an H100 in a cloud serves thousands of tokens per second in total. The site's series explains it from [prefill and decode](/writings/llm-inference-1-prefill-and-decode) to [serving with vLLM](/writings/llm-inference-3-vllm).

### Training and fine-tuning

Training needs much more VRAM than running. The card must also hold the gradients and the optimizer's numbers, which are several times the size of the weights. Fine-tuning a small part of a model, a method called LoRA, needs far less. It works on a 16 GB or 24 GB card for models of a few billion weights. Training a large model from zero is done on racks of H100, B200 or newer GPUs, wired together. The site's piece on [more than one GPU](/writings/llm-inference-7-beyond-one-gpu) covers that wiring.

## What it costs today

> [!WARNING] Read this before the tables
> All prices are in US dollars for the United States, read on **23 November 2026** from public price trackers and reports. GPU prices in 2026 are far above list price and still rising. One quarterly report measured an average of 44% above list, with a rise of about 35% in three months. The reports name two causes, the high price of memory chips and the demand from AI. Trackers disagree with each other, so each cell gives a range. Check a live shop page before you buy.

> [!DEFINITION] List price and street price
> The **street price** is what a shop charges today for a card that is in stock. It is the price you pay, so it comes first in every table here. The **list price** (MSRP) is the price NVIDIA announced at launch. In a shortage the two can be far apart.

<figure class="fig"><svg viewBox="0 0 760 392" role="img" aria-label="A bar chart on a log scale of the street price to buy one NVIDIA product in November 2026, from a Jetson Orin Nano Super kit at 399 dollars to an H100 at about 30,000 dollars."><text x="14" y="22" class="t-title" text-anchor="start">Price to buy one, US dollars, November 2026 (log scale)</text><text x="240" y="54" class="t-note" text-anchor="end">Jetson Orin Nano Super kit</text><rect x="250" y="41" width="41.2" height="18" rx="3" class="s3"/><text x="299.2398148661334" y="54" class="t-val" text-anchor="start">$399</text><text x="240" y="84" class="t-note" text-anchor="end">RTX 5050</text><rect x="250" y="71" width="45.4" height="18" rx="3" class="s1"/><text x="303.3808091037006" y="84" class="t-val" text-anchor="start">$420</text><text x="240" y="114" class="t-note" text-anchor="end">RTX 5060</text><rect x="250" y="101" width="50.6" height="18" rx="3" class="s1"/><text x="308.5911128543729" y="114" class="t-val" text-anchor="start">$448</text><text x="240" y="144" class="t-note" text-anchor="end">RTX 5060 Ti 16 GB</text><rect x="250" y="131" width="93.4" height="18" rx="3" class="s1"/><text x="351.3660005371422" y="144" class="t-val" text-anchor="start">$761</text><text x="240" y="174" class="t-note" text-anchor="end">RTX 5070</text><rect x="250" y="161" width="106.9" height="18" rx="3" class="s1"/><text x="364.90966197674595" y="174" class="t-val" text-anchor="start">$900</text><text x="240" y="204" class="t-note" text-anchor="end">RTX 5070 Ti</text><rect x="250" y="191" width="124.5" height="18" rx="3" class="s1"/><text x="382.49269624911904" y="204" class="t-val" text-anchor="start">$1,119</text><text x="240" y="234" class="t-note" text-anchor="end">RTX 5080</text><rect x="250" y="221" width="171.4" height="18" rx="3" class="s1"/><text x="429.37453615128214" y="234" class="t-val" text-anchor="start">$2,000</text><text x="240" y="264" class="t-note" text-anchor="end">DGX Spark 64 GB</text><rect x="250" y="251" width="245.3" height="18" rx="3" class="s3"/><text x="503.3320856564166" y="264" class="t-val" text-anchor="start">$4,999</text><text x="240" y="294" class="t-note" text-anchor="end">RTX 5090</text><rect x="250" y="281" width="263.4" height="18" rx="3" class="s1"/><text x="521.3629891665908" y="294" class="t-val" text-anchor="start">$6,250</text><text x="240" y="324" class="t-note" text-anchor="end">RTX PRO 6000 Blackwell</text><rect x="250" y="311" width="334.0" height="18" rx="3" class="s2"/><text x="592.0410581001524" y="324" class="t-val" text-anchor="start">$15,000</text><text x="240" y="354" class="t-note" text-anchor="end">H100 or H200 (one card)</text><rect x="250" y="341" width="390.0" height="18" rx="3" class="s2"/><text x="648.0" y="354" class="t-val" text-anchor="start">$30,000</text><text x="14" y="382" class="t-tick" text-anchor="start">Blue: GeForce. Orange: workstation and data center. Green: small computers.</text></svg><figcaption>The street price ladder in November 2026: what one product costs in a US shop today, not its list price. The scale is logarithmic: each step to the right is a multiplication, not an addition.</figcaption></figure>

### GeForce RTX 50 (Blackwell), new

| Card | Memory | Street price, Nov 2026 | List price |
|---|---|---|---|
| RTX 5050 | 8 GB | **about $420** | $249 |
| RTX 5060 | 8 GB | **about $450** | $299 |
| RTX 5060 Ti | 16 GB | **about $760 to $770** | $429 |
| RTX 5070 | 12 GB | **about $900** | $549 |
| RTX 5070 Ti | 16 GB | **about $1,120 to $1,400** | $749 |
| RTX 5080 | 16 GB | **about $2,000** | $999 |
| RTX 5090 | 32 GB | **about $6,250 to $9,400** | $1,999 |

The RTX 5090 row is not a mistake. NVIDIA has not changed its list price of $1,999, but the lowest in-stock price found by different trackers in November runs from $6,249.99 to $9,400. Reports say board makers plan more rises this month.

### Older GeForce cards, used

| Card | Architecture | Memory | Used street price, Nov 2026 | List price at launch |
|---|---|---|---|---|
| RTX 3060 | Ampere | 12 GB | **about $335 to $355** | $329 |
| RTX 4070 | Ada Lovelace | 12 GB | **about $600 to $700** | $599 |
| RTX 3090 | Ampere | 24 GB | **about $1,600 to $1,750** | $1,499 |
| RTX 4090 | Ada Lovelace | 24 GB | **about $2,800 to $3,400** | $1,599 |

A used RTX 3060 costs more today than it did new in 2021. A used RTX 4090 costs about twice its launch price.

### Workstation cards and small AI computers

| Product | Memory | Street price, Nov 2026 | List price at launch |
|---|---|---|---|
| RTX PRO 4000 Blackwell | 24 GB | no current tracker found | $1,546 |
| RTX PRO 4500 Blackwell | 32 GB | no current tracker found | $2,623 |
| RTX PRO 5000 Blackwell | 48 GB | **about $6,000** (tracker of August 2026) | $4,569 |
| RTX PRO 6000 Blackwell | 96 GB | **about $15,000 to $18,000** | $8,565 |
| DGX Spark, 64 GB | 64 GB shared | **$4,999** | new in October 2026 |
| DGX Spark, 128 GB | 128 GB shared | **$6,950** | $3,999 |
| Jetson Orin Nano Super kit | 8 GB shared | **$399** | $249 |
| Jetson AGX Orin kit | 64 GB shared | **$3,499** | not checked |
| Jetson AGX Thor kit | 128 GB shared | **$5,499** | not checked |

For the two cheaper RTX PRO cards I found only the launch prices, so the street price is empty and not guessed. The Jetson AGX prices come from one source.

### Data center GPUs: buy or rent

Few people buy these, so two prices matter: the price of a card and the price of one hour.

| GPU | Architecture | To buy | To rent, per GPU-hour |
|---|---|---|---|
| T4 | Turing | not found | about $0.54 to $0.59 |
| L4 | Ada Lovelace | about $3,750 (used) | about $0.70 to $0.84 |
| A10 | Ampere | not found | about $1.32 |
| L40S | Ada Lovelace | not found | about $1.50 to $1.60 |
| A100 80 GB | Ampere | not found | about $1.94 |
| H100 | Hopper | about $25,000 to $40,000 | median $3.90 |
| H200 | Hopper | about $30,000 to $40,000 | median $4.80 |
| B200 | Blackwell | sold in systems | median $6.90 |
| B300 | Blackwell Ultra | sold in systems | about $7.98 to $9.50 |
| GB300 NVL72 | Blackwell Ultra | a full rack, millions of dollars | about $4.25 to $18.00 |
| Vera Rubin | Rubin | not public | reserved contracts only |

<figure class="fig"><svg viewBox="0 0 760 314" role="img" aria-label="A bar chart on a log scale of cloud rental prices per GPU-hour, from a T4 at about 0.55 dollars to a B300 at about 8 dollars."><text x="14" y="22" class="t-title" text-anchor="start">Price to rent one GPU for one hour, US dollars, late 2026 (log scale)</text><text x="240" y="54" class="t-note" text-anchor="end">T4 (Turing, 2018)</text><rect x="250" y="41" width="62.2" height="18" rx="3" class="s2"/><text x="320.18011592218494" y="54" class="t-val" text-anchor="start">$0.54</text><text x="240" y="84" class="t-note" text-anchor="end">L4 (Ada, 2023)</text><rect x="250" y="71" width="116.0" height="18" rx="3" class="s2"/><text x="373.96209254345587" y="84" class="t-val" text-anchor="start">$0.84</text><text x="240" y="114" class="t-note" text-anchor="end">A10 (Ampere, 2021)</text><rect x="250" y="101" width="171.0" height="18" rx="3" class="s2"/><text x="428.9798639352862" y="114" class="t-val" text-anchor="start">$1.32</text><text x="240" y="144" class="t-note" text-anchor="end">L40S (Ada, 2023)</text><rect x="250" y="131" width="194.4" height="18" rx="3" class="s2"/><text x="452.3962824320775" y="144" class="t-val" text-anchor="start">$1.60</text><text x="240" y="174" class="t-note" text-anchor="end">A100 (Ampere, 2020)</text><rect x="250" y="161" width="217.9" height="18" rx="3" class="s2"/><text x="475.8507339675727" y="174" class="t-val" text-anchor="start">$1.94</text><text x="240" y="204" class="t-note" text-anchor="end">H100 (Hopper, 2022)</text><rect x="250" y="191" width="302.8" height="18" rx="3" class="s2"/><text x="560.8497293883307" y="204" class="t-val" text-anchor="start">$3.90</text><text x="240" y="234" class="t-note" text-anchor="end">H200 (Hopper, 2024)</text><rect x="250" y="221" width="328.1" height="18" rx="3" class="s2"/><text x="586.124576951253" y="234" class="t-val" text-anchor="start">$4.80</text><text x="240" y="264" class="t-note" text-anchor="end">B200 (Blackwell, 2025)</text><rect x="250" y="251" width="372.3" height="18" rx="3" class="s2"/><text x="630.299153590619" y="264" class="t-val" text-anchor="start">$6.90</text><text x="240" y="294" class="t-note" text-anchor="end">B300 (Blackwell Ultra)</text><rect x="250" y="281" width="390.0" height="18" rx="3" class="s2"/><text x="648.0" y="294" class="t-val" text-anchor="start">$7.98</text></svg><figcaption>Renting one GPU for one hour in a cloud. Old architectures stay cheap; each new one costs more per hour and does much more work in that hour.</figcaption></figure>

Rental prices spread widely. For the H100 alone, published offers run from about $1.30 to more than $12 per hour. A small marketplace and a large cloud sell different levels of service. The table gives the median of on-demand offers where an index publishes one.

> [!TIP] Buy or rent
> One H100 at $30,000 equals about 7,700 hours of renting at $3.90, which is 320 days of use without a pause. If you need a large GPU for a few weeks of experiments, rent. If a card will run day and night for years, and you have the power and cooling, buying can cost less.

## Which one is for you

For games, pick by the tier and your screen. The 60 tier is for a 1080p screen, the 70 tier for 1440p and the 80 and 90 tiers for 4K.

For learning AI at home, pick by memory. With 8 GB you can run small models, and an RTX 5050 costs about $420. With 16 GB you can run most open models in compressed form and fine-tune small ones. The cheapest 16 GB card is the RTX 5060 Ti at about $760. With 24 GB or 32 GB you have room for larger models and longer conversations. A used RTX 3090 with 24 GB costs about $1,600 to $1,750. An RTX 5090 with 32 GB costs more than $6,000.

To learn without buying anything, a free Colab notebook gives you a T4 with 16 GB. The site ran a full serving engine on one: [vLLM on a free Colab T4](/writings/llm-inference-8-vllm-on-colab).

For a model that needs more than 32 GB on your own desk, the choices are an RTX PRO card or a DGX Spark. An RTX PRO 6000 with 96 GB costs about $15,000 to $18,000, and a DGX Spark with 128 GB costs $6,950. The Spark has more memory per dollar and is slower. For anything larger, rent an H100, H200 or B200 by the hour.

## What this page cannot tell you

These prices will be out of date soon. They rose by about a third in the three months before this page was written, and nobody knows if they will keep rising or fall. The trackers also measure different things: one reports the lowest in-stock price, another a median of asking prices. I did not buy any of these products, and I did not verify a price on NVIDIA's own shop page.

The memory figure for the B200 depends on the source. NVIDIA's eight-GPU system lists 1,440 GB in total, which is 180 GB for each GPU, and some reports give 192 GB.

> [!TAKEAWAYS]
> - An **architecture** (Ampere, Ada Lovelace, Hopper, Blackwell, Rubin) is a chip design. A **product** (RTX 5070, H100, Jetson Thor) is one size of that design in one package.
> - In a GeForce name, the first digits are the generation and the last two are the tier. In a data center name, the first letter is the architecture.
> - Product families tell you where the GPU lives: GeForce at home, RTX PRO at work, letter cards in servers, Jetson in robots.
> - An **SM** is the block a GPU is built from. A bigger card is the same SM copied more times.
> - For AI, read the **VRAM** first: it decides which models run at all. Then read the **memory bandwidth**: it sets how fast a model writes.
> - **HBM** is the stacked, fast memory of data center cards. Home cards use GDDR.
> - The **street price** is the price you pay. In November 2026 it is far above the list price, so plan with the street price and its date.
