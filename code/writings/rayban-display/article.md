---
title: "How Ray-Ban Display Glasses Put a Screen in a Lens"
description: "How a picture appears inside a clear glasses lens: light, a tiny projector, mirrors hidden in glass, a wristband that reads muscles, and the whole system."
date: 2026-10-24
tags: [optics, smart-glasses, hardware, how-it-works]
motif: waves
accent: "#8fd16a"
---

You are walking to a café. A message arrives: "Running 10 minutes late, order me a coffee?" You do not take out your phone. A small card with the message simply appears in the air, a little to the right of where you are looking. You tap your thumb against your finger, and a reply goes back. The street, the people and the traffic stay in full view the whole time.

That is what the **Meta Ray-Ban Display** glasses do. They went on sale in the United States on 30 September 2025 for $799, a price that includes a wristband called the Meta Neural Band. From the outside they look like a thick pair of Ray-Bans. The right lens looks clear. Yet a sharp, full-colour picture floats inside it.

How can a lens that looks like ordinary glass show a picture? A screen needs pixels and light. A lens has neither, and it sits only about 2 cm from your eye, far too close to read anything. This piece answers the puzzle one step at a time. Each step uses only what the steps before it built. You can start with no physics at all and finish knowing how the whole system works.

> [!TIP] The whole idea in one minute
> A tiny projector sits in the right arm of the frame. It makes the picture on a chip about 2.3 mm wide, lit by tiny LEDs. Lenses then turn each pixel into a beam of parallel light. The beams go into the edge of the right lens, which is really a thin glass plate called a **waveguide**.
>
> Inside the plate the light is trapped. It bounces between the two faces, like a ball rolling down a hallway that bounces off both walls. Hidden in the glass are rows of slanted, half-silvered mirrors. Each mirror sends a little of the light towards your eye and lets the rest continue. Your eye gets many copies of the same picture. Because the light is parallel, the picture seems to float far away, not on the lens.
>
> A wristband reads the tiny electrical signals of your forearm muscles, so a small pinch becomes a click.

{{FIG:overview}}

## Meet the glasses

Before the physics, here is the object itself. Every number below is cited, and the sources are at the end. When a number comes from Meta, it says "Meta says". When it comes from a review, a teardown or a supplier, it says "reported".

| What | Value | Source |
|---|---|---|
| Launch | 30 September 2025, United States, $799 with the Neural Band | Meta newsroom |
| Display | Full colour, right eye only, 600 x 600 pixels | Meta, via Road to VR and UploadVR |
| Field of view | 20 degrees (diagonal) | Meta, via Road to VR; diagonal per Karl Guttag |
| Sharpness | 42 pixels per degree | Meta blog |
| Brightness | 30 to 5,000 nits | Meta, via Road to VR |
| Refresh | 90 Hz display, content updated at 30 Hz | Meta, via Road to VR |
| Light leakage to the outside | 2% | Meta blog |
| Weight | 69 g | Meta blog |
| Battery | Up to 6 hours mixed use; up to 30 hours with the case | Meta newsroom |
| Lenses | Transitions (light-darkening) lenses on all models | Meta blog |
| Prescriptions | -4.00 to +4.00 dioptres | Meta, via UploadVR and Road to VR |
| Camera, audio | 12 MP camera, five microphones, two off-ear speakers | Road to VR |
| Neural Band | sEMG wristband, up to 18 hours, IPX7 water rating | Meta newsroom |

Meta describes the optics in one sentence: "a highly custom light engine and geometric waveguide". Everything else about the optics comes from outside. A teardown by iFixit found the projector chip. The optics analyst Karl Guttag took the projector apart and measured it. The waveguide maker Lumus later said in a press release that Meta "adopted our geometric waveguide technology" for these glasses. We will use all three, and say each time which one a fact comes from.

The words in that table (waveguide, field of view, nits, pixels per degree) are the subject of the rest of this piece. Start with the most basic question of all: what does your eye need to see anything sharp?

## Step 1: what the eye needs

### Light, a lens, and a screen at the back

Your eye is a small camera. Light comes in through a clear window at the front (the cornea) and passes a hole that opens and closes (the pupil). Then it goes through a soft, clear lens. At the back is the **retina**, a sheet of light-sensitive cells. The cornea and the lens bend the light so that every point in the world lands on exactly one point of the retina. That is what "in focus" means.

Think about one point of a distant street lamp. It sends light out in every direction, but only a narrow pencil of it enters your pupil. By the time that pencil reaches you from far away, its rays are almost parallel. The eye's job is to bend those parallel rays so they meet again at one spot on the retina.

Now bring the lamp close. A nearby point sends rays into your pupil that spread apart, because the pupil sees them from a wider angle. Spreading rays are harder to bring together. The eye copes by squeezing its lens into a fatter, stronger shape. This is called **accommodation**, and it has a limit.

> [!DEFINITION] Dioptre
> The strength of a lens, measured as one divided by its focal length in metres. A lens that brings parallel light to a point 0.5 m away has a strength of 2 dioptres. Glasses prescriptions are written in dioptres: -2.00 is a lens of 2 dioptres that spreads light, for short sight.

### Why a screen on the lens cannot work

To focus on something at a distance $$d$$, the eye must add focusing power equal to one over that distance:

$$
P = \frac{1}{d}
$$

Here $$P$$ is the extra power in dioptres, and $$d$$ is the distance to the object in metres. For a street sign 2 m away, $$P = 1/2 = 0.5$$ dioptres, which is easy. For a book at 25 cm, it is 4 dioptres. A young adult's eye can just about manage this; 25 cm is the usual textbook value for the closest comfortable focus.

A display sitting on a glasses lens is about 2 cm from the eye. The eye would need $$P = 1/0.02 = 50$$ dioptres. No human eye can add that. A picture printed on the lens, or a tiny screen glued to it, would be a blur.

{{FIG:eye-focus}}

So the first problem is clear. The glasses must take light from a source very close to the eye and make it look as if it comes from far away. Opticians solved that problem long ago, with a lens.

### A lens can move a picture far away

Put a small picture at exactly the **focal point** of a magnifying lens, and look through the lens. Every point of the picture sends out spreading rays. The lens bends each spray into a bundle of parallel rays. To your eye, parallel rays mean "far away", so it relaxes and sees the picture sharp, as if it were a big poster across the room.

> [!DEFINITION] Collimated light
> Light whose rays all run parallel to each other. A laser pointer beam is nearly collimated. Light from a star is collimated by the time it reaches Earth. A lens turns light from a point at its focal point into collimated light.

> [!DEFINITION] Virtual image
> A picture that the eye sees in a place where no light actually comes from. When you look in a bathroom mirror, your face seems to be behind the glass. Nothing is there; the rays only look as if they started there. The floating screen of display glasses is a virtual image.

There is a second, quieter trick in the same picture. Each point of the display leaves the lens as a bundle at its own angle. The top of the display becomes rays tilted one way, and the bottom becomes rays tilted the other way. **Position on the display has become direction.** Hold on to this idea. It is the reason the rest of the system can work at all.

{{FIG:virtual-image}}

The thin-lens rule makes this exact. A lens with focal length $$f$$ takes an object at distance $$d_o$$ in front of it and makes an image at distance $$d_i$$:

$$
\frac{1}{d_i} = \frac{1}{f} - \frac{1}{d_o}
$$

Read it as "the bending power of the lens, minus the spread of the incoming light, gives the convergence of the outgoing light". With $$f = 10$$ mm and the display exactly at $$d_o = 10$$ mm, the right side is zero, so the image is at infinity. Move the display to 9.9 mm and the numbers in our script give a virtual image about 0.99 m in front of the eye. At 9.5 mm the image comes to 0.19 m.

A shift of a tenth of a millimetre moves the picture by a metre, which shows how precise these optics must be. (These are example values to show the rule; Meta does not publish the distance at which the Ray-Ban Display picture appears.)

<details><summary>Research notes: why "far away" is not "infinitely far away"</summary>

Display makers often set the virtual image at a finite distance, such as one or two metres, rather than at infinity. A picture at infinity is relaxing for the eye. But text you glance at while talking to someone nearby can feel more natural a little closer. Each choice also changes how the picture lines up with the real world for the other eye. Meta has not published the image distance of the Ray-Ban Display. The thin-lens numbers above come from `numbers.py`, section 2.

</details>

So now we know what to build: a tiny bright picture, and optics that make its light parallel. But one thing is still wrong. A magnifier must sit right in front of the eye. In glasses, the space in front of the eye is taken by the lens you look through. The light must come from the side, travel across the lens, and turn towards the eye, while the lens stays clear. That needs the physics of light inside glass.

## Step 2: how light behaves in glass

### Rays, reflection and the normal

For everything in this piece, light can be treated as **rays**: straight lines that carry energy. A ray changes direction only where it meets a surface between two materials. At that surface two things can happen. Some light bounces back (reflection), and some goes through but changes direction (refraction).

> [!DEFINITION] Normal
> An imaginary line at right angles to a surface, at the point where a ray hits it. Optics measures every angle from the normal, not from the surface. A ray that hits a window head-on has an angle of 0 degrees.

Reflection follows the rule you know from a game of pool. The angle going out equals the angle coming in, on the other side of the normal.

### Refraction and the refractive index

Light travels more slowly in glass than in air. Imagine a line of marching students walking from a paved square onto a muddy field at a slant. The students who reach the mud first slow down first, so the whole line swings round. Light does the same at a surface. The part of the wave that enters the glass first slows first, and the ray turns towards the normal.

> [!DEFINITION] Refractive index (n)
> How much slower light travels in a material than in empty space. Air has n of almost exactly 1. Water has about 1.33, common glass and acrylic about 1.5, and special high-index materials go higher. A larger n means slower light and more bending.

The rule that gives the new angle is **Snell's law**:

$$
n_1 \sin\theta_1 = n_2 \sin\theta_2
$$

Read it left to right. $$n_1$$ is the refractive index of the material the ray starts in, and $$\theta_1$$ is its angle from the normal there. $$n_2$$ and $$\theta_2$$ are the same two things in the material it enters. The product of index and sine stays the same across the surface. The slower side must have the smaller angle.

A tiny example. A ray in air ($$n_1 = 1$$) hits glass ($$n_2 = 1.5$$) at $$\theta_1 = 30°$$. The sine of 30° is 0.5. So $$1 \times 0.5 = 1.5 \times \sin\theta_2$$, which gives $$\sin\theta_2 = 0.333$$ and $$\theta_2 = 19.5°$$. The ray turns towards the normal, from 30° to 19.5°.

{{FIG:refraction}}

<figure class="narrow"><img src="/img/writings/rayban-display/img-tir-block.jpg" alt="A half-round block of clear acrylic on a dark table. A thin beam of light enters the curved side, and at the flat face part of it leaves while part of it reflects back into the block." loading="lazy" /><figcaption>A beam of light meeting the flat face of a half-round acrylic block. Part of it bends out into the air and part reflects back. Photo: Friedrich Graf, CC BY-SA 3.0, via Wikimedia Commons.</figcaption></figure>

### Total internal reflection: light that cannot get out

Now run the same law backwards, from glass out into air. A ray inside the glass at 30° gives $$1.5 \times 0.5 = 0.75 = \sin\theta_2$$, so it leaves at 48.6°. Going from slow to fast, the ray bends away from the normal.

Tilt the ray more. At 45° inside the glass, $$1.5 \times \sin 45° = 1.061$$. But the sine of an angle can never be larger than 1, so there is no angle in the air that satisfies the law. No ray can leave. All of the light reflects back into the glass, with no loss at all at a clean surface. This is **total internal reflection**, often shortened to TIR.

> [!DEFINITION] Total internal reflection (TIR)
> Light inside a slow material (higher n) can meet the surface with a faster one (lower n) at a steep angle. Then none of it gets out, and all of it reflects. This is how optical fibres carry internet traffic under oceans, and how the glasses carry the picture across the lens.

The boundary is the **critical angle**. It is the angle at which the escaping ray would just skim along the surface, at 90° in the air:

$$
\sin\theta_c = \frac{n_2}{n_1}
$$

Here $$\theta_c$$ is the critical angle inside the denser material, $$n_1$$ is that material's index and $$n_2$$ is the index outside. For glass against air, $$\sin\theta_c = 1/1.5 = 0.667$$, so $$\theta_c = 41.8°$$. Any ray inside the glass steeper than 41.8° from the normal is trapped.

{{FIG:tir}}

The higher the index, the smaller the critical angle, and the wider the range of angles that stay trapped. That range is the "room" a waveguide has for the picture, which is why material makers chase high index. Meta wrote in 2025 that the best glass it had in 2016 had an index of 1.8. Its Orion prototype instead uses silicon carbide, with an index of 2.7.

{{FIG:critical}}

<figure class="narrow"><img src="/img/writings/rayban-display/img-tir-plate.jpg" alt="A red laser beam inside a flat sheet of clear acrylic, zigzagging between the two faces of the sheet as it travels along it." loading="lazy" /><figcaption>A red laser beam trapped in an acrylic sheet by total internal reflection. It zigzags between the two faces; specks in the plastic scatter a little light so you can see the path. Photo: KlausFoehl, CC BY-SA 3.0, via Wikimedia Commons.</figcaption></figure>

### A plate that carries light

That laser photo is the heart of the glasses. A flat plate of glass, with light entering at a steep angle, becomes a pipe for light. The light walks along the plate in zigzags. How far it moves along the plate between two touches of the same face depends only on the thickness $$t$$ and the angle $$\theta$$:

$$
\text{stride} = 2\,t\,\tan\theta
$$

$$t$$ is the plate thickness and $$\theta$$ is the ray's angle from the normal. With an example 1 mm plate and a ray at 60°, the stride is $$2 \times 1 \times 1.732 = 3.46$$ mm. Over 30 mm of lens, the light makes about 8.7 round trips. (The thickness of the Ray-Ban Display waveguide is not public. These are example values.)

{{FIG:zigzag}}

Now we have every idea we need for the rest. A lens makes a tiny picture look far away. Each pixel becomes a direction. Glass can trap light by total internal reflection and carry it sideways. The glasses join these three ideas. The first part of the machine is the thing that makes the picture.

## Step 3: the tiny projector

### Four ways to make a very small picture

The picture starts on a **microdisplay**: a display about the size of a fingernail or smaller, with pixels a few micrometres across. A microdisplay pixel is far thinner than a human hair.

> [!DEFINITION] Microdisplay
> A very small, very dense display made to be magnified by optics, not looked at directly. Camera viewfinders, projectors and head-worn displays use them. The picture is usually under 1 cm across.

There are four main families. Two of them make their own light; one reflects light from a lamp; one draws the picture with a single beam.

{{FIG:microdisplays}}

**LCoS** (liquid crystal on silicon) is a mirror chip. Each pixel is a tiny mirror under a thin layer of liquid crystal. It controls how much of the incoming light comes back in a usable form. It needs a separate light source, usually LEDs.

**Micro-OLED** builds organic light-emitting pixels straight onto a silicon chip, so each pixel glows and black pixels are truly off. **MicroLED** makes each pixel a tiny inorganic LED, which can be very bright. Making millions of them small and in three colours is still hard. **Laser scanning** uses one beam of light steered by a fast moving mirror to draw the picture line by line.

<figure><img src="/img/writings/rayban-display/img-microled.jpg" alt="Microscope images of a microLED microdisplay: a grid of tiny round LED pixels with a 12 micrometre scale, a diagram of the LED layers bonded onto a silicon driver chip, and a green picture of a leopard shown on the 9.6 mm display." loading="lazy" /><figcaption>An early active-driven microLED microdisplay: 640 x 480 pixels, each 12 micrometres across, bonded to a silicon driver chip, showing a green picture 9.6 mm wide. Image: Santasgift, from work by Day, Li, Lie, Bradford, Lin and Jiang. CC BY-SA 4.0, via Wikimedia Commons.</figcaption></figure>

### What the Ray-Ban Display uses

Meta has not said which microdisplay it uses. The iFixit teardown found an **OmniVision OP03010 LCoS panel** in the projector, with a grid of 600 x 600 pixels and three LEDs. Karl Guttag reports that it is the same device as OmniVision's catalogue part OP03011, on a different flexible cable. Meta chose LCoS here, while its Orion prototype uses microLED projectors, according to Road to VR's report of Meta's 2024 announcement.

OmniVision's product brief for the OP03011 gives the size. The chip has 648 x 648 pixels at a pitch of 3.8 micrometres, so the active area is 2.462 mm on each side. If the glasses use 600 x 600 of them, as the specs say, the picture on the chip is $$600 \times 3.8\ \mu\text{m} = 2.28$$ mm wide. The whole picture you see starts on a square about 2.3 mm wide.

<figure class="narrow"><img src="/img/writings/rayban-display/img-lcos-coin.jpg" alt="A small rectangular microdisplay module on a ribbon cable, next to a stack of circuit boards and a 1 euro coin that is larger than the display window." loading="lazy" /><figcaption>A reflective LCoS microdisplay (Forth Dimension Displays WXGA-R5) next to a 1 euro coin. This is not the chip in the Ray-Ban Display, which is smaller still. Photo: Forth Dimension Displays, CC BY-SA 4.0, via Wikimedia Commons.</figcaption></figure>

### How an LCoS pixel makes light or dark

To follow LCoS, you need one more property of light: polarisation.

> [!DEFINITION] Polarisation
> Light is a wave, and the wave shakes in a direction across the beam. If it shakes only up and down, the light is vertically polarised; only side to side, horizontally polarised. Polarised sunglasses block one of these directions, which is why they cut glare.

> [!DEFINITION] Liquid crystal
> A material whose long molecules flow like a liquid but line up like a crystal. Lined-up molecules can twist the polarisation of light that passes through them. An electric voltage tilts the molecules and changes how much they twist it. Every LCD screen works this way.

Inside the chip, under a cover glass, is a thin layer of liquid crystal. Under that, each pixel has a polished aluminium mirror, and under the mirror sits the silicon circuit that sets its voltage. Polarised light comes in, passes the liquid crystal, bounces off the pixel mirror and passes the liquid crystal again on the way out.

If a pixel's voltage leaves its polarisation unchanged, the light comes back the way it went in. If the voltage makes the crystal turn the polarisation by 90 degrees over the round trip, it comes back turned. Values in between give grey levels. The chip itself makes no picture you could see: it only paints the picture into the polarisation of the light.

{{FIG:lcos}}

A **polarising beam splitter** turns that hidden picture into a visible one. It is a glass cube with a special coating on its diagonal that reflects one polarisation and passes the other. It sends the LED light onto the chip, and when the light comes back, it passes only the turned light onwards. Turned pixels become bright, unturned pixels become dark.

> [!DEFINITION] Polarising beam splitter (PBS)
> A cube or plate that separates light by polarisation: one direction passes straight through, the other is reflected sideways. In an LCoS projector it acts both as the doorway to the chip and as the filter that turns polarisation into brightness.

### Colour from one chip: field-sequential colour

An LCD in a laptop has three sub-pixels (red, green, blue) at every point. A 3.8 micrometre LCoS pixel has no room for that. Instead, the chip shows the colours one after the other. It shows the red part of the picture while only the red LED is on. Then it shows the green part with the green LED, then the blue part.

Your eye blends the quick flashes into one full-colour picture. OmniVision's brief says the OP03011 is "designed for color field sequential operation".

> [!DEFINITION] Field-sequential colour
> Making a colour picture by showing its red, green and blue parts one after another, fast enough that the eye merges them. It saves space on the chip but can show brief rainbow edges ("colour break-up") when your eyes move quickly.

Road to VR lists a 90 Hz display refresh with content refreshing at 30 Hz. At 90 frames a second with three colour fields each, the chip must show at least 270 colour fields every second. Each field lasts at most 3.7 ms. Guttag notes that the chip itself can run up to 120 frames a second, and that colour break-up is visible on the glasses.

{{FIG:fsc}}

### The light engine

Guttag's teardown describes the whole projector, which Goertek reportedly designed. It is an unusual layout. The red and blue LEDs sit side by side on one ceramic base, so only one colour-combining mirror is needed. A pair of tiny lens arrays (a "fly-eye" homogeniser) spreads the light evenly. The polarising beam splitter sends it to the LCoS chip. A curved mirror and lenses then collimate the returning picture and send it into the waveguide.

> [!DEFINITION] Dichroic mirror
> A mirror with a thin-film coating that reflects some colours and passes others. Two LEDs of different colours can be merged into one beam with it.

{{FIG:engine}}

At full brightness, Guttag reports that a colleague measured about **1 lumen** of light coming out of this projector. That is very little light. Remember this number; it will matter when we follow where the light goes. Meta calls the whole unit "a highly custom light engine", and that is as much as Meta has said about it.

## Step 4: the waveguide

### The job: in, along, out

At the exit of the projector we have what the eye needs. Each pixel is a bundle of parallel light at its own angle. The waveguide has three jobs. It must take that light in at the edge of the lens (**in-coupling**) and carry it across without spoiling the picture (**propagation by TIR**). Then it must send it out towards the eye (**out-coupling**). It must do all of this while you see the world clearly through it.

> [!DEFINITION] Waveguide
> In smart glasses, a thin, clear plate that carries light inside itself by total internal reflection. It lets the light out where it is needed. You look straight through it at the world.

**In-coupling** is the easy part to picture. A mirror at an angle near the edge of the plate turns the incoming beams. Now they hit the faces at steep angles, past the critical angle, and they are trapped. A pixel at the top of the picture and one at the bottom enter at slightly different angles. They keep that difference all the way across.

Why does that matter? Because a flat mirror changes the direction of all rays in the same way. A ray that enters 3 degrees above another still leaves 3 degrees above it, after any number of bounces off flat, parallel faces. The plate keeps the picture, because the picture lives in the angles. This is why "position becomes direction" was so important.

{{FIG:waveguide}}

### Out-coupling with partial mirrors

To get the light out, a **geometric waveguide** (also called a reflective waveguide) has a row of slanted, partly reflecting mirrors buried in the glass. When the trapped light passes through the first one, a small share of it reflects at a new angle. That angle is steep enough to escape through the face of the plate towards the eye. The rest passes through and travels on to the next mirror, and so on.

> [!DEFINITION] Partial mirror
> A surface coated to reflect only part of the light that hits it and pass the rest, like one-way glass in a police interview room. A bare glass surface already reflects about 4% of the light at head-on incidence.

These mirrors are so thin and so clear from your side that you look through them at the world. Lumus, whose design Meta adopted, has worked on this idea since its early patents. The drawing below is Figure 2 of a Lumus patent from 2004. Light from a display (18) enters at the right and reflects off an in-coupling mirror (16). It zigzags along the plate (20) and meets partial mirrors (22) that send it to the eye (24).

<figure class="fig paper"><img src="/img/writings/rayban-display/pat-fig2.png" alt="Patent drawing of a long thin plate. Light enters from below at the right, reflects off an angled mirror, zigzags along the plate, and meets a row of slanted partial mirrors that each send parallel rays down to a drawn eye." loading="lazy" /><figcaption>Figure 2 of US patent 6,829,095 B2, "Substrate-guided optical beam expander" (Yaakov Amitai, Lumus, 2004), turned on its side. US patents are in the public domain.</figcaption></figure>

### Sharing the light fairly

There is a catch. The first mirror sees all the light, but the last mirror sees only what the others let through. If every mirror reflected the same share, the picture would be bright near the projector and dim on the far side. So each mirror must reflect a larger share than the one before it.

The same patent gives the rule, for mirrors that spread the light sideways in a second direction:

> [!PAPER] Amitai (2004), US patent 6,829,095 · column 11
> [![A paragraph of the patent text, with 33%, 50% and 100% highlighted, describing three reflecting surfaces whose reflectances are chosen to give uniform output](/img/writings/rayban-display/p-patent-ladder.png)](/img/writings/rayban-display/p-patent-ladder.png)
>
> **What it says:** "if three reflecting surfaces are used, then the reflectances are set to be 33%, 50% and 100%" for the first, second and third surface, "so as to achieve uniform output waves".
>
> **What it means:**
>
> Picture three friends sharing a pizza that is passed down a line. The first friend takes a third of the whole pizza. The second sees two thirds left and takes half of it, which is again a third of the whole. The last friend takes everything that is left: the last third. Each gets the same amount, but each takes a bigger share of what reaches them.
>
> The mirrors do the same with light. The general rule for $$N$$ mirrors is that mirror number $$k$$, counting from 1, reflects $$1/(N-k+1)$$ of what reaches it. Six mirrors would need 17%, 20%, 25%, 33%, 50% and 100%.
>
> A 100% mirror is opaque, so it would block your view of the world. Real see-through designs use low reflectances and accept that some light goes to waste. The same patent gives a second example: 20%, 25% and 33%. Each mirror sends 20% to the eye, and 40% of the light is left over at the end.

{{FIG:ladder}}

### The eyebox: why the picture is copied many times

Why have many mirrors at all? Why not one mirror right in front of the eye? The answer is that your eye moves, and so do the glasses on your nose.

The beam that comes out of a small projector is only a few millimetres wide. Your pupil is also a few millimetres wide: about 1.5 to 4 mm in daylight. With only one exit beam, the picture would vanish as soon as your eye rolled a little. It would also vanish if the frame slid down your nose, or sat on a face with wider-set eyes. Each partial mirror sends out its own copy of the beam, and side by side the copies fill a wider box of light.

> [!DEFINITION] Eyebox
> The region behind the lens where an eye can be and still see the whole picture. A larger eyebox makes glasses easier to fit and more forgiving of eye movement. Meta has not published the eyebox size of the Ray-Ban Display.

> [!DEFINITION] Pupil expansion
> Making the exit beam of a display wider by copying it many times side by side. The name comes from the "exit pupil", the opening that the light of a projector leaves through.

The copies do not make a confusing jumble, because of the same trick as before. Each pixel is a direction, not a place. Every copy of the beam carries every pixel at its own angle. Wherever your pupil sits in the eyebox, it catches some copy, and inside that copy each pixel arrives from its correct direction. The picture stays in the same spot in your view.

{{FIG:eyebox}}

The patent's Figure 9 shows the same idea from the eye's side. Three groups of rays (solid, dashed and dotted lines) are three directions, so three pixels. Each leaves the plate from several mirrors, and all of them reach the one eye.

<figure class="fig paper narrow"><img src="/img/writings/rayban-display/pat-fig9.png" alt="Patent drawing of a flat plate with a row of slanted partial mirrors. Three sets of rays, drawn as solid, dashed and dash-dot lines, leave the plate at three angles from many mirrors and all converge on one eye below." loading="lazy" /><figcaption>Figure 9 of US patent 6,829,095 B2 (Amitai, Lumus, 2004). Each line style is one direction, so one pixel; each leaves from many mirrors and all reach the eye. Public domain.</figcaption></figure>

### Copying in two directions

A single row of mirrors widens the beam in one direction only, say left to right. The eye also moves up and down. So the light must be copied in two directions. The 2004 patent already sketches a second set of angled mirrors (its Figure 13 below). It spreads the light sideways before the main set spreads it down. Later Lumus patents describe "multiple-axis" expansion in more compact forms.

<figure class="fig paper narrow"><img src="/img/writings/rayban-display/pat-fig13.png" alt="Patent drawing seen from the front: a box marked 20 sends light up through three angled mirrors 22a, 22b and 22c, which send it sideways towards a row of six long vertical mirrors marked 22." loading="lazy" /><figcaption>Figure 13 of US patent 6,829,095 B2 (Amitai, Lumus, 2004). Three angled mirrors (22a to 22c) copy the beam from the display (20) along one direction; the long mirrors (22) then copy it again along the other. Public domain.</figcaption></figure>

<figure class="fig paper narrow"><img src="/img/writings/rayban-display/pat-2d.png" alt="Patent drawing of a clear plate in perspective. A beam enters at the top, is copied by a first set of slanted reflectors into several parallel beams, and these are copied again and leave the plate towards an eye on the left." loading="lazy" /><figcaption>Figure 1 of US patent 10,551,544 B2, "Light-guide optical element with multiple-axis internal aperture expansion" (Danziger and Sharlin, Lumus, 2020). The beam is copied along two directions inside one plate. Public domain.</figcaption></figure>

Guttag judges the Ray-Ban Display waveguide to be "clearly a variation on Lumus's Z-Lens Geometric waveguide", though he could not see the usual Z-Lens mirror stripes. Neither Meta nor Lumus has published the exact layout, so the sketch below shows the principle, not the real mirror counts.

{{FIG:expand2d}}

### The other way: diffraction gratings

Many smart glasses with a picture in the lens, including Microsoft's HoloLens headsets and Meta's own Orion prototype, use a different kind of waveguide. Instead of buried mirrors, they use **diffraction gratings**: patterns of ridges so fine that their spacing is close to the wavelength of light. Light that meets a grating splits into a few beams at new angles, much like the rainbow you see on the back of a CD.

> [!DEFINITION] Diffraction grating
> A surface with a regular pattern of very fine lines or ridges. It sends light of each wavelength into a set of definite directions. Different colours go to different angles, which is why gratings make rainbows.

> [!DEFINITION] Diffractive waveguide
> A waveguide that uses gratings, usually stamped or etched on the surface of the plate, to couple light in and out. It is flat and easy to copy in large numbers, but it treats each colour differently.

The rule for a grating, for light arriving head-on and bent into the glass, is the **grating equation**:

$$
n \sin\theta = \frac{\lambda}{\Lambda}
$$

Here $$n$$ is the refractive index of the glass and $$\theta$$ is the angle of the bent ray inside it. $$\lambda$$ (lambda) is the wavelength of the light, and $$\Lambda$$ (capital lambda) is the spacing of the grating lines. Longer wavelengths bend more.

Take an example grating with a 400 nm spacing on glass with $$n = 1.5$$. Blue light at 450 nm gives $$\sin\theta = 450/600 = 0.75$$, so it enters at 48.6°. Green at 530 nm enters at 62.0°. Both are past the critical angle of 41.8°, so both are trapped. Red at 630 nm gives $$\sin\theta = 1.05$$, which is impossible: red has no ray inside the glass at all.

{{FIG:grating}}

This tiny example shows the core weakness of diffractive designs. One grating cannot treat three colours alike. Designers work around it with two or three stacked plates, one per colour range, careful grating shapes, and high-index materials. Meta's 2025 article on Orion says its silicon carbide waveguides remove the "rainbows" that ambient light makes on a grating. A mirror, by contrast, reflects every colour at the same angle, so a geometric waveguide carries full colour through one plate.

<figure class="narrow"><img src="/img/writings/rayban-display/img-hololens.jpg" alt="NASA astronaut Scott Kelly inside the International Space Station, wearing a Microsoft HoloLens headset with a dark visor over his eyes." loading="lazy" /><figcaption>Astronaut Scott Kelly tries a Microsoft HoloLens, a headset with diffractive waveguides, on the International Space Station in February 2016. Photo: NASA, public domain, via Wikimedia Commons.</figcaption></figure>

### Light that leaks out of the front

A waveguide does not send light only to the eye. A grating bends light both ways, so some of it leaves through the front of the lens, towards the people around you. In some glasses you can see a glowing patch on the wearer's lens. This is called **eye glow**, and it is a privacy problem as much as a looks problem: someone facing you might read your messages.

> [!DEFINITION] Light leakage (eye glow)
> Light from a see-through display that leaves through the front of the lens instead of going to the eye. It wastes power, and it can let others see that the display is on, or even what it shows.

Meta says the Ray-Ban Display has only 2% light leakage. Guttag measured the forward glow at about 1.5% of the brightness reaching the eye, directed downwards. He contrasts it with 50% to 100% for typical diffractive waveguides. At full brightness, 1.5% of 5,000 nits is about 75 nits leaving the front, aimed at the ground.

{{FIG:geo-diff}}

| | Geometric (reflective) | Diffractive |
|---|---|---|
| How light gets out | Buried partial mirrors | Surface gratings |
| Colour | All colours alike, one plate | Each colour differs; often two or three plates |
| Forward leak | Low (about 1.5% measured here) | Often high (50 to 100%, per Guttag) |
| Rainbows from room light | Few | A known problem |
| Efficiency | Lumus-type designs "typically 3 to 7 times" better, per Guttag | Lower |
| Manufacturing | Many coated slices, cut and glued; hard to scale | Stamping or etching; scales well |
| Examples | Meta Ray-Ban Display (Lumus design, reported to be made by Schott) | HoloLens, Orion prototype |

The last row has a supply story. Schott, the German glass maker, said shortly before the launch that it was the "first company capable of handling geometric reflective waveguide manufacturing in production volumes". iFixit calls Schott and Lumus "almost certain" to be the supplier. In January 2026, Meta paused its planned launch in four more countries, citing "unprecedented demand and limited inventory".

### How little of the light reaches you

Now we can put the projector's one lumen together with the 5,000 nits at the eye. First, what is a nit?

> [!DEFINITION] Luminance and the nit
> Luminance is how bright a surface looks: the light it sends towards you per unit of area, per unit of viewing angle. It is measured in candelas per square metre, also called nits. A typical computer monitor peaks around 250 nits. A clear sky averages around 8,000 nits.

A small amount of light, spread over a small area and a small cone of directions, can still look very bright. The light that enters your pupil is:

$$
\Phi = L \times A \times \Omega
$$

$$\Phi$$ (phi) is the light flux in lumens. $$L$$ is the luminance in nits. $$A$$ is the pupil area in square metres. $$\Omega$$ (omega) is the solid angle of the picture, the size of the cone of directions it fills, in steradians.

Our script works it out for the real numbers. The picture spans about 14.1° x 14.1° (we derive that in Step 5), a solid angle of 0.0606 steradians. A 3 mm pupil has an area of 7.07 square millimetres. At 5,000 nits, $$\Phi = 5000 \times 7.07 \times 10^{-6} \times 0.0606 = 0.00214$$ lumens. That is **2.14 millilumens**, or about **0.21%** of the 1 lumen that leaves the projector.

{{FIG:budget}}

Where does the other 99.8% go? Most of it lights the rest of the eyebox, the places your pupil could be but is not. More is lost at partial mirrors that send light the wrong way and at the edges of the plate. A little more leaks forward. This is the price of pupil expansion: the system lights a whole window so that your small pupil can find it anywhere. It is also why efficiency matters so much in glasses that run on a tiny battery.

## Step 5: seeing the world and the picture at once

### Contrast against a bright world

A see-through display has a problem a phone does not have. It cannot make black. A "black" pixel simply sends no light, and you see the world through it. The display can only add light on top of the world. So what matters is not the display's brightness alone, but how it compares with what is behind it.

> [!DEFINITION] Contrast ratio
> How much brighter the brightest part of a picture is than the darkest. For a see-through display it is (display + world) divided by (world): the picture's white over the world it sits on.

$$
C = \frac{L_d + T\,L_w}{T\,L_w}
$$

$$L_d$$ is the display luminance at the eye. $$L_w$$ is the luminance of the world behind the picture, and $$T$$ is the fraction of world light that the lens lets through. A contrast of 1 means the picture is invisible. The higher the contrast, the easier the picture is to read.

Try it at the brightest setting. A 5,000-nit picture on an average clear sky of 8,000 nits, through a clear lens, gives $$C = (5000 + 8000)/8000 = 1.62$$. That is weak but readable. A display with the 250 nits of a typical monitor would give $$C = 1.03$$: on a sunny day it would be all but invisible. This is why Meta made the display so bright. The range goes from 30 nits for a dark room up to 5,000 for the street.

{{FIG:nits}}

### The darkening lens

All Meta Ray-Ban Display glasses ship with Transitions lenses, which darken in sunlight. This is not just a style choice. Look at the contrast formula again. Dimming the world (making $$T$$ smaller) raises the contrast, as long as the display light is not dimmed with it.

> [!DEFINITION] Photochromic lens
> A lens that darkens when ultraviolet light hits it and clears again indoors. The molecules in the lens change shape under UV and absorb more visible light. Transitions is a well-known brand of them.

<figure><img src="/img/writings/rayban-display/img-photochromic.jpg" alt="The same pair of wraparound cycling glasses photographed twice. On the left, at dusk by the sea, the lenses are almost clear. On the right, in bright sun in a field, the lenses are dark grey." loading="lazy" /><figcaption>The same photochromic lens at dusk (left, nearly clear) and in sunlight (right, dark). Photos: JaredMcKenzie, CC0, via Wikimedia Commons.</figcaption></figure>

A lens dark enough to count as typical sunglasses (European filter category 3, which lets 8% to 18% through) changes the numbers a lot. With the lens at 18%, the sky behind the picture drops to 1,440 nits, and the contrast rises to 4.47. At 8%, it is 8.81. The darkening lens is part of the display system.

{{FIG:contrast}}

### The lens sandwich

The trick works only if the darkening part sits on the world side of the waveguide. The iFixit teardown found lenses glued on both sides of the waveguide, and one of them darkens. Light from the world passes the outer lens, the waveguide and the inner lens. Light from the display starts inside the waveguide, so it passes only the inner lens on its way to the eye. The darkening dims the street but not the message.

{{FIG:stack}}

The inner lens is also where a prescription goes. UploadVR reports that Meta supports short sight, long sight and astigmatism from -4 to +4 dioptres of total power. Up to 4 dioptres of cylinder correct astigmatism. A 4-dioptre lens has a focal length of 25 cm. The prescription is bonded to the lens during manufacturing, at a cost UploadVR reports as $200. You cannot simply pop in new lenses later.

> [!DEFINITION] Astigmatism (cylinder)
> An eye whose cornea is curved more in one direction than another, like the back of a spoon instead of a ball. It blurs lines in some directions more than others. A "cylinder" value in a prescription corrects it.

The order of the sandwich matters for the picture too. The display light leaves the waveguide as parallel rays, which look far away. A short-sighted wearer's inner lens spreads that light a little, so the far picture comes into focus. Their normal glasses do exactly the same for a far street sign. The picture is corrected along with the world.

### One eye, and how the brain copes

The Ray-Ban Display shows its picture to the right eye only. Displays like this are called **monocular**. The left eye sees only the world.

> [!DEFINITION] Monocular display
> A display shown to one eye only. A binocular display shows a picture to both eyes, which allows depth (3D). It needs two projectors and two waveguides that agree very precisely.

Your brain is used to merging two slightly different views into one. Here one eye sees a small card and the other does not, yet most people perceive one view with a card in it. Some find that it takes a little getting used to. UploadVR calls the picture "a small fixed HUD", a head-up display that stays put as you turn, rather than graphics fixed to objects in the room.

Meta has not explained the choice of one eye. The likely reasons follow from everything above. A second display means a second projector, a second waveguide, more weight and more battery drain. Two pictures must also line up to a small fraction of a degree, or the eyes strain to fuse them. And glasses flex on a head. One display avoids that problem entirely.

### Field of view and pixels per degree

How big is the picture? Optics measures size as an angle. A small picture close up and a big one far away look the same if they fill the same angle.

> [!DEFINITION] Field of view (FOV)
> The angle that the picture fills in your view, measured across it. The Ray-Ban Display's field of view is 20 degrees, measured from corner to corner (the diagonal).

A square picture 20 degrees across the diagonal is $$20/\sqrt{2} = 14.14$$ degrees on each side. At a reading distance of 40 cm, that is the size of a square about 9.9 cm wide. (This uses the small-angle approximation, which is close at these sizes.)

> [!DEFINITION] Arcminute
> One sixtieth of a degree. A person with normal (20/20) vision can just pick out detail about 1 arcminute in size.

> [!DEFINITION] Pixels per degree (ppd)
> How many pixels fit into one degree of your view. It measures sharpness the way the eye does, whatever the screen's real size. About 60 ppd matches the limit of normal vision.

The sharpness then follows from two specs:

$$
\text{ppd} = \frac{\text{pixels across}}{\text{degrees across}}
$$

Along the diagonal there are $$600 \times \sqrt{2} = 848.5$$ pixels, over 20 degrees. That gives $$848.5 / 20 = 42.4$$ pixels per degree, which matches Meta's figure of 42. One pixel spans $$60 / 42.4 = 1.41$$ arcminutes. Normal vision resolves about 1 arcminute, so you could just see the pixels on a perfect display. To reach 60 ppd over the same square, it would need about 849 x 849 pixels.

{{FIG:ppd}}

The optics do not deliver every pixel perfectly. Guttag estimates that the picture resolves "closer to 400x400 pixels", which would be about 28 effective pixels per degree. He also says that in most use the display shows about 16 degrees or less. That is an outside estimate, not a Meta figure, but it is a reminder that a spec sheet number is a ceiling. On brightness, Guttag says he could confirm Meta's figure of about 5,000 nits.

### Why so small, and why off to the side

Twenty degrees is a small window. Your two eyes together take in a view many times wider. Why not a bigger picture?

Every part of the system gets harder with field of view. The waveguide must trap a wider range of angles, which needs a higher refractive index. The pixel count grows with the square of the width. At the same 42 ppd, a square 70 degrees on the diagonal would need about 2,079 x 2,079 pixels: 12 times as many. The light needed to fill a bigger picture at the same brightness grows with its solid angle. Each step costs size, heat and battery.

{{FIG:fov}}

Meta says it placed the display "off to the side, so it doesn't obstruct your view". A card in the corner of your vision is easy to glance at and easy to ignore. A picture in the centre would block the very street you are walking along. Together with the small size, this placement makes the display a place to check, not a place to live, which suits messages, directions and captions.

## Step 6: the hand as the input

### A screen with no touchscreen

You can speak to Meta AI through the glasses, but talking aloud is not always possible or polite. Meta's answer is the **Meta Neural Band**, a fabric wristband worn on the writing hand. It turns small finger movements into commands, even with the hand resting at your side.

Meta's help page lists the gestures. A tap of the index finger on the thumb selects. A tap of the middle finger on the thumb goes back. A swipe of the thumb along the side of the index finger scrolls, like swiping a phone. Holding the index finger to the thumb and turning the wrist changes the volume or zooms. A double tap of the thumb on the side of the index finger calls Meta AI.

### Muscles are electrical

To move a finger, your brain sends a signal down your spinal cord to **motor neurons**. Each motor neuron connects to a group of muscle fibres in your forearm, which move your fingers through long tendons. When a motor neuron fires, all its fibres fire together, and each fibre makes a tiny electrical pulse as it contracts.

> [!DEFINITION] Motor unit
> One motor neuron plus all the muscle fibres it controls. When it fires, they all contract together. A muscle is made of many motor units; a stronger movement fires more of them, more often.

> [!DEFINITION] Electromyography (EMG) and surface EMG (sEMG)
> Recording the electrical activity of muscles. Clinical EMG often uses a needle in the muscle. Surface EMG uses electrodes on the skin above the muscle, so it needs no needle at all.

The pulses of many fibres add up and spread through the tissue to the skin, where electrodes can pick them up. They are tiny: Figure 1 of the paper below uses scale bars of 10 and 400 microvolts (millionths of a volt). The spike that one motor unit makes is called a motor-unit action potential. It is not a brain signal: the wristband listens to the muscles, at the end of the chain.

<figure><img src="/img/writings/rayban-display/img-emg-lab.jpg" alt="A researcher at a desk looks at a large monitor showing red electrical traces with bursts of spikes, recorded from leg muscles during walking." loading="lazy" /><figcaption>Surface EMG in a biomechanics laboratory at the University of Ottawa: bursts of muscle activity recorded while a person stops walking. Photo: D. Gordon E. Robertson, CC BY-SA 3.0, via Wikimedia Commons.</figcaption></figure>

{{FIG:emg-chain}}

### The research behind the band

Meta's Reality Labs published the science behind the band in *Nature* in 2025. The paper describes a research wristband and the models that decode it, tested on people who had never used it before.

> [!PAPER] Kaifosh, Reardon et al. (2025), A generic non-invasive neuromotor interface for human-computer interaction · Abstract · page 702
> [![The abstract of the Nature paper, with generic sEMG decoding models that generalize across people and handwriting at 20.9 words per minute highlighted](/img/writings/rayban-display/p-emg-abstract.png)](/img/writings/rayban-display/p-emg-abstract.png)
>
> **What it says:** the team built a wristband and data collection from "thousands of consenting participants", which let them "develop generic sEMG decoding models that generalize across people". New users reached "handwriting at 20.9 words per minute".
>
> **What it means:**
>
> The hard part of EMG was never recording it. It was that every arm is different. Wrists differ in size, muscles sit in slightly different places, and the band never lands in exactly the same spot twice. A model trained on one person usually fails on the next, so older EMG systems needed a long calibration for each user.
>
> The word "generic" is the breakthrough. By recording thousands of people, the team trained models that work for a new person straight out of the box. Speech recognition works the same way for voices it has never heard. The 20.9 words per minute is for writing letters with a finger on a surface, with no pen and no screen.

The excerpts from this paper come from the open-access article, which is licensed CC BY-NC-ND 4.0.

The wristband itself is a careful piece of engineering. It must pick up signals of a few microvolts through dry skin, with no gel, while you move.

> [!PAPER] Kaifosh, Reardon et al. (2025), A generic non-invasive neuromotor interface for human-computer interaction · Scalable sEMG recording platform · page 703
> [![A paragraph of the paper describing the sEMG research device, with multichannel recording device with a high sample rate and low-noise highlighted](/img/writings/rayban-display/p-emg-device.png)](/img/writings/rayban-display/p-emg-device.png)
>
> **What it says:** the research device is "a dry electrode, multichannel recording device with a high sample rate (2 kHz) and low-noise (2.46 μVrms)", made in four sizes.
>
> **What it means:**
>
> "Dry electrode" means metal pins touch the skin directly, with no sticky gel pad. That is what lets you put the band on in seconds. "Multichannel" means many pairs of pins around the wrist, each listening to the muscles beneath it.
>
> A sample rate of 2 kHz means each channel is measured 2,000 times a second. With 16 channels, that is 32,000 numbers every second for the model to read. The noise of 2.46 microvolts is the electrical "hiss" of the band itself, and it must sit below the faint signals of single motor units.

The paper's Figure 1 shows the whole setup. You see a participant with the band, the prompts on screen and the electrodes around a slice of the wrist. You also see raw signals from all the channels during gestures.

> [!PAPER] Kaifosh, Reardon et al. (2025), A generic non-invasive neuromotor interface for human-computer interaction · Figure 1 · page 704
> [![Figure 1 of the Nature paper: a person wearing the wristband at a laptop with gesture prompts; the band with electrode pins and a compute capsule; an MRI slice of the wrist with electrode positions; three task screens; and colourful multi-channel EMG traces during pinches and swipes](/img/writings/rayban-display/p-emg-fig1.png)](/img/writings/rayban-display/p-emg-fig1.png)
>
> **What it says:** the band "consists of 48 electrode pins configured into 16 bipolar channels", with a compute capsule that "digitizes the signal and streams sEMG data using Bluetooth".
>
> **What it means:**
>
> Look at the wrist slice in panel b. The coloured dots around it are the electrode positions. They sit over the flexor muscles (that close the fingers) on one side and the extensors (that open them) on the other. A gap sits over the ulna bone, where there is little muscle to hear.
>
> "Bipolar" means each channel measures the difference between two neighbouring pins. Noise that reaches both pins equally, such as mains hum, cancels out. Panel d shows why the problem is both hard and possible. Each gesture lights up its own pattern across the channels, but the patterns are buried in busy signals.

### From signals to a gesture

A network reads the 16 channels as a stream over time. For discrete gestures, the paper uses a 1D convolution layer, three LSTM layers and a classifier. For handwriting it uses frequency-power features and a conformer, a model that mixes attention with convolution.

> [!DEFINITION] LSTM
> Long short-term memory: a kind of neural network that reads a sequence one step at a time and keeps a running memory. It suits signals where what happened a moment ago matters, like the start and end of a pinch.

> [!DEFINITION] Conformer
> A neural network that combines convolution (good at local patterns) with attention (good at linking distant moments). It was first built for speech recognition, a task much like reading handwriting from a stream of signals.

{{FIG:decoder}}

The models are honest about the cost of a band versus a hand-held device. The paper compares each task with a baseline:

> [!PAPER] Kaifosh, Reardon et al. (2025), A generic non-invasive neuromotor interface for human-computer interaction · Online evaluation of generic models · page 707
> [![A paragraph comparing the sEMG decoders with a trackpad, a game controller and handwriting without a decoder, with at 25.1 WPM, higher than the 20.9 WPM achieved with the sEMG handwriting decoder highlighted](/img/writings/rayban-display/p-emg-baseline.png)](/img/writings/rayban-display/p-emg-baseline.png)
>
> **What it says:** writing on a surface with no decoder runs "at 25.1 WPM, higher than the 20.9 WPM achieved with the sEMG handwriting decoder", and a phone keyboard reaches 36 WPM. The decoders "have room to improve" but "reliably complete each task".
>
> **What it means:**
>
> The band's handwriting reaches about 83% of the speed at which the same people write with a finger and no decoder. A phone keyboard is still faster. The point is not to beat the phone. It is to type a short reply without one, hands at your sides, while you walk.
>
> The same paragraph gives other baselines. A trackpad acquires targets faster than the wrist decoder, and a game controller completes grid moves faster than the gesture decoder. Every number in it is from people using the decoder for the first time, with no training on their own data.

The product goes further than the paper. Meta says the Neural Band's models were trained on data from nearly 200,000 consenting research participants. In January 2026, Meta began an early-access test of "neural handwriting" for WhatsApp and Messenger, limited at first to the United States and to English. The paper's research device recorded at 2 kHz with a battery life of more than 4 hours; Meta says the product band lasts up to 18 hours.

## Step 7: the whole system

### What is inside the frame

Now step back and see the glasses as one machine. iFixit's teardown found the parts. The left arm holds the battery, the open-ear speakers and microphones. The right arm holds a Qualcomm Snapdragon AR1 chip with 2 GB of memory and 32 GB of storage, next to the projector. Road to VR lists a 12-megapixel camera and five microphones in total.

{{FIG:system}}

The glasses connect without wires to your phone and its Meta AI app, and through the phone to the internet. Meta says the display shows Meta AI answers with pictures, and messages from WhatsApp, Messenger and Instagram. It also shows live video calls and a camera viewfinder with zoom. In supported cities it gives walking directions with a map, and it shows live captions, translation and music controls.

<figure><img src="/img/writings/rayban-display/img-glass-evolution.png" alt="Two line drawings of glasses with displays. On the left, a frame with a bulky projector and prism block beside the right eye. On the right, ordinary-looking glasses with small display areas inside both lenses." loading="lazy" /><figcaption>An older approach and a newer one, drawn from two Google patents: a prism display that sits in front of the lens (left), and displays built into the lenses (right). Drawing: Michael Weinold, CC BY 4.0, via Wikimedia Commons.</figcaption></figure>

### The power budget

The battery is small. iFixit reports a 960 milliwatt-hour battery, glued into the left arm. Meta says the glasses last up to six hours of mixed use. Divide the two:

$$
\bar{P} = \frac{E}{t} = \frac{960\ \text{mWh}}{6\ \text{h}} = 160\ \text{mW}
$$

$$\bar{P}$$ is the average power, $$E$$ the stored energy, and $$t$$ the run time. The glasses average about 160 milliwatts over a day of mixed use, much of it with the display off. At a steady 1 watt, the same battery would last about an hour. (The two numbers come from different sources, so this is an estimate.)

> [!DEFINITION] Milliwatt-hour (mWh)
> A unit of stored energy. A battery of 960 mWh can supply 960 milliwatts for one hour, or 160 milliwatts for six hours.

That budget explains many design choices we met. A display that sends only 0.21% of its light to your pupil must be efficient everywhere else. A geometric waveguide that Guttag calls 3 to 7 times more efficient than a diffractive one saves battery directly. Field-sequential LCoS needs only three small LEDs. A monocular display needs one projector, not two. And because everything sits on your face, every milliwatt also becomes heat near your skin.

### The weight

Meta says the glasses weigh 69 grams. Every part competes for that budget: battery, chip, projector, waveguide, speakers, frame and lenses. A heavier frame is tiring on the nose, and a larger battery means a thicker arm. Most trade-offs in this piece are one trade-off seen from different angles. The goal is a picture that is good enough, in glasses that people will wear all day.

## Step 8: limits, and what comes next

### What the glasses do not do

The picture is a small, fixed window. It is not augmented reality in the full sense: graphics do not stick to objects in the room, and there is no depth. UploadVR put it plainly at launch: "not AR". The colours can break up when your eyes move fast, because of field-sequential colour. The picture is sharp by the spec sheet, softer by Guttag's estimate.

The prescription range of -4 to +4 dioptres leaves out people with stronger prescriptions. And the glasses are only as useful as the apps that draw on the display.

### What changed in 2026

At CES in January 2026, Meta announced a teleprompter feature and early access to neural handwriting. It also paused its planned international launch because of limited stock.

In September 2026, at Connect, Meta added online ordering in the United States. It opened sales in Canada and the UK, with France, Italy and Germany from 13 October 2026. Road to VR reports prices from £749, CA$1,149 and €899. Meta also announced cycling and transit directions, a photorealistic "hologram" of your face for video calls, and an Explore app. No new display hardware was announced.

### Wider fields of view, in the lab

Two lines of work point to bigger pictures. Meta's **Orion** glasses, shown in 2024, are a prototype, not a product. They use silicon carbide waveguides, microLED projectors and a field of view of about 70 degrees, according to Meta and Road to VR.

At CES 2026, Lumus showed **ZOE**, made from standard optical glass. Lumus calls it the first geometric waveguide with a field of view over 70 degrees. It also showed an updated 30-degree module, the Z-30. These are waveguide demonstrations by a supplier, not products you can buy, and no one has announced which glasses will use them.

> [!WARNING] Products versus prototypes
> Everything in Steps 1 to 7 is about the Meta Ray-Ban Display you can buy. Orion and ZOE are a prototype and a demo. Claims about future products are plans until they ship.

The path forward is visible in the physics. Wider pictures need materials with a higher refractive index, so more angles stay trapped. They need more pixels, so microdisplays with smaller pixels. They need more light, so more efficient waveguides and emitters. True AR, with graphics fixed in the room, needs two displays that agree with each other. It also needs cameras that track the world, all inside about the same weight.

> [!TAKEAWAYS]
> - The eye cannot focus on anything 2 cm away. Glasses displays make the light parallel, so the picture looks far away (a virtual image).
> - A lens turns position on a display into direction. Every later trick relies on that: the picture lives in the angles of the light.
> - Glass traps light past the critical angle (41.8° for n = 1.5) by total internal reflection. So a thin plate can carry the picture across the lens.
> - The projector is reported to be an OmniVision LCoS chip, 600 x 600 pixels of 3.8 micrometres, lit by LEDs one colour at a time.
> - A geometric waveguide (Lumus design, which Lumus confirms) uses buried partial mirrors. Each sends a share of the light to the eye, copying the beam to fill an eyebox.
> - Mirrors treat all colours alike and leak little light forward: Meta says 2%; Guttag measured about 1.5%. Diffractive waveguides often leak far more.
> - Only about 0.21% of the projector's light enters a 3 mm pupil. The rest fills the eyebox or is lost, so efficiency is everything.
> - See-through contrast needs brightness: 5,000 nits against an 8,000-nit sky gives only 1.62 : 1. Darkening lenses on the world side raise it to 4.5 to 8.8 : 1.
> - 600 pixels across 20 degrees diagonal gives 42.4 pixels per degree, below the 60 that matches normal vision.
> - The Neural Band reads muscle signals at the wrist with surface EMG. Models trained on thousands of people work for a new wearer without calibration.

## References

- Meta (2025). [Meta Ray-Ban Display: Breakthrough AI Glasses Available Now](https://www.meta.com/blog/meta-ray-ban-display-ai-glasses-connect-2025/) (42 ppd, 2% leakage, 69 g, geometric waveguide).
- Meta Newsroom (2025). [Meta Ray-Ban Display and Meta Neural Band](https://about.fb.com/news/2025/09/meta-ray-ban-display-ai-glasses-emg-wristband/) (price, launch, battery, features).
- Road to VR (2025). [Meta Ray-Ban Display specs](https://roadtovr.com/meta-ray-ban-smart-glasses-display-price-release-date-specs/) (resolution, brightness range, refresh, camera, microphones).
- UploadVR (2025). [Meta Ray-Ban Display officially announced](https://www.uploadvr.com/meta-ray-ban-display-glasses-officially-announced/) and [prescription support](https://www.uploadvr.com/meta-ray-ban-display-glasses-prescription-support/).
- iFixit (2025). [Teardown: groundbreaking waveguide tech inside Meta's $800 AR glasses](https://www.ifixit.com/News/113543/theres-groundbreaking-waveguide-tech-inside-metas-800-ar-glasses-but-dont-count-on-fixing-them).
- Karl Guttag (2025). [Meta Ray-Ban Display Part 1: Lumus waveguide, OmniVision LCOS, and Goertek projection engine](https://kguttag.com/2025/10/30/meta-ray-ban-display-part-1-lumus-waveguide-omnivision-lcos-and-goertek-projection-engine/).
- OmniVision (2023). [OP03011 LCOS product brief](https://www.ovt.com/wp-content/uploads/2023/05/OP03011-PB-v1.0-WEB.pdf).
- Lumus (2026). [Next-generation waveguides at CES 2026](https://www.prnewswire.com/news-releases/lumus-unveils-next-gen-waveguides-for-ar-glasses-at-ces-2026-including-its-first-geometric-waveguide-to-exceed-70-field-of-view-302653598.html).
- Patents: [US 6,829,095 B2](https://patents.google.com/patent/US6829095B2/en) (Amitai, Lumus, 2004) and [US 10,551,544 B2](https://patents.google.com/patent/US10551544B2/en) (Danziger and Sharlin, Lumus, 2020).
- Meta (2025). [Silicon carbide waveguides and the path to Orion's large field of view](https://www.meta.com/blog/orion-silicon-carbide-waveguides-ar-glasses-large-field-of-view/); Road to VR (2024), [Orion prototype](https://roadtovr.com/meta-orion-prototype-reveal-connect-2024/).
- Kaifosh, Reardon et al. (2025). [A generic non-invasive neuromotor interface for human-computer interaction](https://doi.org/10.1038/s41586-025-09255-w), *Nature* 645, 702 to 711. Open access, CC BY-NC-ND 4.0.
- Meta. [How to use Meta Neural Band](https://www.meta.com/help/ai-glasses/764536076119235/) (gestures).
- Meta (2026). [CES 2026: teleprompter and neural handwriting](https://www.meta.com/blog/ces-2026-meta-ray-ban-display-teleprompter-emg-handwriting-garmin-unified-cabin-university-of-utah-tetraski/); Meta Newsroom (2026), [new features at Connect 2026](https://about.fb.com/news/2026/09/new-features-for-meta-ray-ban-display-navigation-hologram/); Road to VR (2026), [international launch](https://roadtovr.com/meta-ray-ban-display-int-launch-connect-2026/).
- Background values from Wikipedia: [orders of magnitude of luminance](https://en.wikipedia.org/wiki/Orders_of_magnitude_%28luminance%29), [total internal reflection](https://en.wikipedia.org/wiki/Total_internal_reflection), [sunglass filter categories](https://en.wikipedia.org/wiki/Sunglasses), [pupil size](https://en.wikipedia.org/wiki/Pupil), [near point](https://en.wikipedia.org/wiki/Near_point), [visual acuity](https://en.wikipedia.org/wiki/Visual_acuity).
- Our code: `code/writings/rayban-display/` (`numbers.py` and its saved output, figures, screenshots).
