"""Every Apple M-series chip /machine can draw: what it is and how its dies
are laid out.

``CHIPS`` is the lineup as verified 2026-11-14 against Apple's newsroom and
Wikipedia. cpu: (kind, cores), kind S super, P performance, E efficiency
(M1-M4's performance cores are the role M5 calls super; they keep the name
Apple gave them then). gpu: physical cores of the top config, and the binned
ones. bw: GB/s as Apple states it, top and binned.

``PLANS`` are the floorplans, traced from Apple's own images, never guessed.
Every block is (kind, x0, x1, y0, y1, opts) in the pixel coordinates of a
600 px view of the Apple image it came from; T() turns them into fractions
of the die. Sources (apple.com/newsroom, viewed 2026-11-14):

  M1              Apple_new-m1-chip_11102020 (die render)
  M1 Pro / Max    Apple-M1-chip-family-lineup-220308 (die renders, to scale)
  M1 Ultra        same lineup image: two M1 Max dies, the top one mirrored
  M2              Apple-WWDC22-M2-SoC-220606 (die render)
  M2 Pro / Max    Apple-M2-chips-M2-Pro / -M2-Max-230117 (die in the package)
  M2 / M3 Ultra   no die image from Apple (badge only): two Max dies joined
                  the way Apple's M1 Ultra render shows
  M3 / Pro / Max  Apple-M3-chip-series-architecture-231030 (die renders)
  M4 / Pro / Max  Apple-M4-chip-new-CPU-240507, Apple-M4-Pro/Max-chip-CPU-
                  performance (block diagrams)
  M5              Apple-M5-chip-251015 (block diagram)
  M5 Pro / Max /  no die image from Apple (badges only). Built from Apple's
  Ultra, M6       stated parts (Fusion Architecture: a CPU die and a GPU die;
                  M5 Ultra two of those pairs) in the M5 diagram's block style.

Kinds: gpu, s (super), p (performance; mid=True for M5's smaller P cores),
e (efficiency), ne (neural engine), sram, phy, logic. Apple's logo and
lettering regions are drawn as plain logic: no text, no logo. Stdlib only.
"""
CHIPS = {
    # name: cpu, gpu (top, binned), ne cores, bw (top, binned), memory, nm, transistors B
    "M1":       ((("P", 4), ("E", 4)), (8, (7,)), 16, (68, None), "LPDDR4X", 5, 16),
    "M1 Pro":   ((("P", 8), ("E", 2)), (16, (14,)), 16, (200, None), "LPDDR5", 5, 33.7),
    "M1 Max":   ((("P", 8), ("E", 2)), (32, (24,)), 16, (400, None), "LPDDR5", 5, 57),
    "M1 Ultra": ((("P", 16), ("E", 4)), (64, (48,)), 32, (800, None), "LPDDR5", 5, 114),
    "M2":       ((("P", 4), ("E", 4)), (10, (8, 9)), 16, (100, None), "LPDDR5", 5, 20),
    "M2 Pro":   ((("P", 8), ("E", 4)), (19, (16,)), 16, (200, None), "LPDDR5", 5, 40),
    "M2 Max":   ((("P", 8), ("E", 4)), (38, (30,)), 16, (400, None), "LPDDR5", 5, 67),
    "M2 Ultra": ((("P", 16), ("E", 8)), (76, (60,)), 32, (800, None), "LPDDR5", 5, 134),
    "M3":       ((("P", 4), ("E", 4)), (10, (8,)), 16, (100, None), "LPDDR5", 3, 25),
    "M3 Pro":   ((("P", 6), ("E", 6)), (18, (14,)), 16, (150, None), "LPDDR5", 3, 37),
    "M3 Max":   ((("P", 12), ("E", 4)), (40, (30,)), 16, (400, 300), "LPDDR5", 3, 92),
    "M3 Ultra": ((("P", 24), ("E", 8)), (80, (60,)), 32, (819, None), "LPDDR5", 3, 184),
    "M4":       ((("P", 4), ("E", 6)), (10, (8, 9)), 16, (120, None), "LPDDR5X", 3, 28),
    "M4 Pro":   ((("P", 10), ("E", 4)), (20, (16,)), 16, (273, None), "LPDDR5X", 3, None),
    "M4 Max":   ((("P", 12), ("E", 4)), (40, (32,)), 16, (546, 410), "LPDDR5X", 3, None),
    "M5":       ((("S", 4), ("E", 6)), (10, (8,)), 16, (153, None), "LPDDR5X", 3, None),
    "M5 Pro":   ((("S", 6), ("P", 12)), (20, (16,)), 16, (307, None), "LPDDR5X", 3, None),
    "M5 Max":   ((("S", 6), ("P", 12)), (40, (32,)), 16, (614, 460), "LPDDR5X", 3, None),
    "M5 Ultra": ((("S", 12), ("P", 24)), (80, (64,)), 32, (1228, None), "LPDDR5X", 3, None),
    "M6":       ((("S", 2), ("P", 4), ("E", 6)), (12, (10,)), 32, (153, None), "LPDDR5X", 2, None),
}
# No M4 Ultra (skipped). No M6 Pro, Max or Ultra.


def bandwidth(name, gpu_cores=None):
    """GB/s for ``name`` with ``gpu_cores`` on: the binned figure for a binned chip."""
    top, binned = CHIPS[name][3]
    return binned if binned and gpu_cores and gpu_cores < CHIPS[name][1][0] else top



def T(box, size, blocks):
    X0, X1, Y0, Y1 = box
    W, H = X1 - X0, Y1 - Y0
    out = []
    for b in blocks:
        kind, x0, x1, y0, y1 = b[:5]
        o = b[5] if len(b) > 5 else {}
        out.append((kind, (x0 - X0) / W, (y0 - Y0) / H, (x1 - x0) / W, (y1 - y0) / H, o))
    return {"size": size, "blocks": out}


def L(*boxes):
    """Plain logic. Slivers under 14 px of the trace are left as bare die:
    drawn, they read as stray lines."""
    return [("logic",) + b for b in boxes if min(b[1] - b[0], b[3] - b[2]) >= 14]


# -- M1 ----------------------------------------------------------------------------

M1 = T((30, 585, 30, 555), (11.2, 10.6), [
    ("gpu", 30, 420, 30, 245, {"cores": 8, "cols": 4}),
    ("p", 420, 585, 30, 245, {"cores": 4, "cols": 2}),
    ("e", 30, 175, 245, 440, {"cores": 4, "cols": 2}),
    ("sram", 175, 295, 245, 305, {"banks": 4}),
    ("ne", 175, 295, 305, 440, {"cores": 16, "cols": 2}),
    ("sram", 350, 410, 245, 300, {"banks": 2}),
    ("sram", 350, 410, 340, 395, {"banks": 2}),
    ("phy", 30, 300, 500, 555, {"chans": 8}),
] + L((295, 350, 245, 500), (410, 585, 245, 340), (410, 585, 340, 500),
      (350, 410, 300, 340), (350, 410, 395, 500), (30, 175, 440, 500),
      (175, 295, 440, 500), (300, 460, 500, 555), (460, 585, 500, 555)))

M1_PRO = T((25, 585, 110, 490), (19.0, 12.9), [
    ("gpu", 150, 460, 340, 490, {"cores": 16, "cols": 8}),
    ("sram", 80, 150, 340, 490, {"banks": 6, "vertical": True}),
    ("sram", 460, 530, 340, 490, {"banks": 6, "vertical": True}),
    ("phy", 25, 80, 270, 490, {"chans": 6}),
    ("phy", 530, 585, 270, 490, {"chans": 6}),
    ("p", 185, 385, 165, 335, {"cores": 8, "cols": 4}),
    ("e", 385, 480, 110, 200, {"cores": 2, "l2": "bottom"}),
    ("ne", 385, 585, 200, 270, {"cores": 16, "cols": 8}),
] + L((25, 185, 110, 270), (185, 385, 110, 165), (480, 585, 110, 200),
      (385, 530, 270, 340), (80, 185, 270, 340)))

M1_MAX = T((15, 500, 20, 590), (19.2, 22.5), [
    ("phy", 15, 500, 20, 60, {"chans": 10}),
    ("phy", 15, 65, 60, 445, {"chans": 8}),
    ("phy", 455, 500, 60, 590, {"chans": 10}),
    ("sram", 65, 125, 115, 385, {"banks": 6, "vertical": True}),
    ("sram", 390, 455, 115, 385, {"banks": 6, "vertical": True}),
    ("gpu", 125, 390, 115, 385, {"cores": 32, "cols": 8}),
    ("ne", 15, 190, 445, 515, {"cores": 16, "cols": 8}),
    ("p", 190, 360, 395, 500, {"cores": 8, "cols": 4}),
    ("e", 360, 455, 395, 470, {"cores": 2, "l2": "bottom"}),
] + L((65, 455, 60, 115), (65, 190, 385, 445), (125, 390, 385, 395),
      (360, 455, 470, 520), (15, 345, 515, 590), (345, 455, 520, 590), (190, 360, 500, 515)))

# -- M2 ----------------------------------------------------------------------------

M2 = T((45, 555, 50, 510), (13.1, 11.8), [
    ("gpu", 45, 455, 50, 195, {"cores": 10, "cols": 5}),
    ("sram", 455, 555, 50, 260, {"banks": 6}),
    ("p", 45, 175, 195, 420, {"cores": 4, "cols": 2}),
    ("e", 175, 310, 195, 290, {"cores": 4, "cols": 4, "rows": 1, "l2": "bottom"}),
    ("ne", 175, 350, 340, 420, {"cores": 16, "cols": 8}),
    ("sram", 350, 420, 260, 340, {"banks": 3}),
    ("sram", 500, 555, 260, 420, {"banks": 4, "vertical": True}),
    ("phy", 45, 300, 420, 510, {"chans": 8}),
] + L((310, 455, 195, 260), (175, 350, 290, 340), (420, 500, 260, 420),
      (350, 420, 340, 420), (300, 555, 420, 510)))

M2_PRO = T((25, 595, 100, 488), (20.6, 14.0), [
    ("sram", 105, 160, 195, 245, {"banks": 2}),
    ("p", 210, 390, 145, 320, {"cores": 8, "cols": 4}),
    ("e", 390, 545, 100, 200, {"cores": 4, "cols": 4, "rows": 1, "l2": "bottom"}),
    ("sram", 430, 545, 265, 320, {"banks": 4}),
    ("phy", 25, 75, 330, 488, {"chans": 6}),
    ("phy", 545, 595, 270, 488, {"chans": 6}),
    ("sram", 75, 140, 330, 488, {"banks": 6, "vertical": True}),
    ("sram", 480, 545, 320, 488, {"banks": 6, "vertical": True}),
    ("gpu", 140, 480, 330, 488, {"cores": 20, "cols": 10}),
] + L((25, 105, 100, 330), (105, 210, 100, 195), (105, 210, 245, 330), (160, 210, 195, 245),
      (210, 390, 100, 145), (210, 390, 320, 330), (545, 595, 100, 270), (390, 545, 200, 265),
      (390, 430, 265, 330), (430, 480, 320, 330)))

M2_MAX = T((22, 552, 30, 572), (22.3, 22.8), [
    ("p", 195, 385, 75, 230, {"cores": 8, "cols": 4}),
    ("e", 385, 552, 75, 180, {"cores": 4, "cols": 4, "rows": 1, "l2": "bottom"}),
    ("sram", 385, 552, 180, 235, {"banks": 6}),
    ("phy", 22, 75, 235, 572, {"chans": 10}),
    ("phy", 500, 552, 235, 572, {"chans": 10}),
    ("sram", 75, 130, 240, 465, {"banks": 6, "vertical": True}),
    ("sram", 440, 500, 240, 465, {"banks": 6, "vertical": True}),
    ("gpu", 130, 440, 240, 465, {"cores": 40, "cols": 10}),
    ("ne", 235, 335, 465, 560, {"cores": 16, "cols": 4}),
] + L((22, 552, 30, 75), (22, 195, 75, 235), (195, 385, 230, 240),
      (75, 235, 465, 572), (335, 500, 465, 572), (235, 335, 560, 572)))

# -- M3 ----------------------------------------------------------------------------

M3 = T((15, 585, 18, 500), (13.1, 11.1), [
    ("gpu", 15, 585, 18, 180, {"cores": 10, "cols": 5}),
    ("p", 15, 200, 180, 330, {"cores": 4, "rows": 1, "l2": "bottom", "l2frac": 0.4}),
    ("ne", 200, 355, 180, 265, {"cores": 16, "cols": 8}),
    ("e", 15, 230, 330, 410, {"cores": 4, "rows": 1, "l2": "bottom", "l2frac": 0.25}),
    ("sram", 530, 585, 330, 500, {"banks": 4, "vertical": True}),
    ("phy", 120, 530, 410, 500, {"chans": 6}),
] + L((355, 585, 180, 270), (200, 255, 265, 330), (255, 530, 270, 410),
      (530, 585, 270, 330), (15, 120, 410, 500), (230, 255, 330, 410)))

M3_PRO = T((15, 500, 10, 595), (14.3, 17.3), [
    ("gpu", 15, 500, 10, 255, {"cores": 18, "cols": 6}),
    ("sram", 170, 355, 255, 285, {"banks": 4}),
    ("p", 45, 205, 345, 450, {"cores": 6, "cols": 3}),
    ("e", 45, 205, 450, 540, {"cores": 6, "cols": 3}),
    ("sram", 210, 380, 350, 395, {"banks": 2}),
    ("phy", 15, 500, 540, 595, {"chans": 8}),
] + L((15, 170, 255, 345), (355, 500, 255, 340), (170, 355, 285, 345), (15, 45, 345, 540),
      (205, 380, 345, 350), (210, 380, 395, 540), (380, 500, 340, 540), (205, 210, 350, 540)))

M3_MAX = T((10, 565, 15, 590), (23.6, 24.4), [
    ("p", 135, 420, 65, 205, {"cores": 12, "cols": 6}),
    ("e", 420, 495, 65, 200, {"cores": 4, "cols": 2}),
    ("ne", 60, 520, 210, 262, {"cores": 16, "cols": 8}),
    ("phy", 10, 60, 170, 590, {"chans": 10}),
    ("phy", 520, 565, 170, 590, {"chans": 10}),
    ("sram", 60, 110, 265, 590, {"banks": 6, "vertical": True}),
    ("sram", 465, 520, 265, 590, {"banks": 6, "vertical": True}),
    ("gpu", 110, 465, 265, 590, {"cores": 40, "cols": 8}),
] + L((10, 565, 15, 65), (10, 135, 65, 170), (495, 565, 65, 170), (60, 135, 170, 210),
      (420, 520, 200, 210), (135, 420, 205, 210)))

# -- M4 (Apple's block diagrams) ---------------------------------------------------

M4 = T((18, 450, 18, 452), (12.8, 12.9), [
    ("p", 18, 228, 88, 285, {"cores": 4, "rows": 1, "l2": "none"}),
    ("e", 18, 228, 285, 357, {"cores": 6, "cols": 3, "l2": "none"}),
    ("ne", 18, 228, 357, 425, {"cores": 16, "cols": 8}),
    ("gpu", 228, 450, 18, 225, {"cores": 10, "cols": 5}),
    ("sram", 340, 450, 285, 452, {"banks": 6}),
] + L((18, 228, 18, 88), (18, 228, 425, 452), (228, 303, 225, 285), (303, 378, 225, 285),
      (378, 450, 225, 285), (228, 340, 285, 335), (228, 340, 335, 425), (228, 340, 425, 452)))

M4_PRO = T((12, 535, 12, 428), (18.4, 14.7), [
    ("p", 12, 185, 65, 338, {"cores": 10, "cols": 5, "l2": "none"}),
    ("e", 12, 185, 338, 428, {"cores": 4, "cols": 2, "l2": "none"}),
    ("gpu", 185, 535, 12, 250, {"cores": 20, "cols": 10}),
    ("ne", 185, 325, 250, 428, {"cores": 16, "cols": 4}),
    ("sram", 465, 535, 295, 405, {"banks": 4}),
] + L((12, 185, 12, 65), (325, 395, 250, 295), (395, 465, 250, 295), (465, 535, 250, 295),
      (325, 395, 295, 340), (395, 430, 295, 340), (430, 465, 295, 405), (325, 430, 340, 405),
      (325, 535, 405, 428)))

M4_MAX = T((8, 556, 8, 595), (21.6, 23.2), [
    ("p", 8, 198, 62, 498, {"cores": 12, "cols": 4, "rows": 3, "l2": "none"}),
    ("e", 8, 198, 498, 595, {"cores": 4, "cols": 2, "l2": "none"}),
    ("gpu", 198, 556, 8, 408, {"cores": 40, "cols": 10}),
    ("ne", 198, 318, 408, 595, {"cores": 16, "cols": 4}),
    ("sram", 480, 556, 455, 560, {"banks": 4}),
] + L((8, 198, 8, 62), (318, 398, 408, 455), (398, 478, 408, 455), (478, 556, 408, 455),
      (318, 420, 455, 560), (420, 480, 455, 560), (318, 556, 560, 595)))

# -- M5 (Apple's block diagram) ----------------------------------------------------

M5 = T((100, 500, 52, 452), (12.8, 12.8), [
    ("s", 100, 298, 122, 300, {"cores": 4, "rows": 1, "l2": "none"}),
    ("e", 100, 298, 300, 372, {"cores": 6, "cols": 2, "rows": 3, "l2": "none"}),
    ("ne", 100, 298, 372, 452, {"cores": 16, "cols": 8}),
    ("gpu", 298, 500, 52, 300, {"cores": 10, "cols": 5}),
    ("sram", 378, 500, 300, 452, {"banks": 6}),
] + L((100, 298, 52, 122), (298, 378, 300, 335), (298, 378, 335, 410), (298, 378, 410, 452)))

# M5 Pro / Max / Ultra and M6: Apple has shown no die. Same 0-100 frame, the
# M5 diagram's blocks, Apple's stated core counts.
M5_CPU = T((0, 100, 0, 100), (11.0, 13.2), [
    ("s", 0, 100, 10, 40, {"cores": 6, "cols": 6, "rows": 1, "l2": "none"}),
    ("p", 0, 100, 40, 70, {"cores": 12, "cols": 6, "mid": True, "l2": "none"}),
    ("ne", 0, 60, 70, 100, {"cores": 16, "cols": 8}),
    ("sram", 60, 100, 70, 100, {"banks": 4}),
] + L((0, 50, 0, 10), (50, 100, 0, 10)))

M5_GPU20 = T((0, 100, 0, 100), (15.0, 13.2), [
    ("gpu", 0, 100, 0, 62, {"cores": 20, "cols": 10}),
    ("sram", 40, 100, 62, 88, {"banks": 6}),
    ("phy", 0, 100, 88, 100, {"chans": 8}),
] + L((0, 20, 62, 88), (20, 40, 62, 88)))

M5_GPU40 = T((0, 100, 0, 100), (15.0, 22.0), [
    ("gpu", 0, 100, 0, 72, {"cores": 40, "cols": 10}),
    ("sram", 40, 100, 72, 90, {"banks": 6}),
    ("phy", 0, 100, 90, 100, {"chans": 10}),
] + L((0, 20, 72, 90), (20, 40, 72, 90)))

M6 = T((0, 100, 0, 100), (12.8, 12.8), [
    ("s", 0, 50, 17, 42, {"cores": 2, "rows": 1, "l2": "none"}),
    ("p", 0, 50, 42, 62, {"cores": 4, "rows": 1, "mid": True, "l2": "none"}),
    ("e", 0, 50, 62, 80, {"cores": 6, "cols": 3, "l2": "none"}),
    ("ne", 0, 50, 80, 100, {"cores": 32, "cols": 16}),
    ("gpu", 50, 100, 0, 62, {"cores": 12, "cols": 6}),
    ("sram", 70, 100, 62, 100, {"banks": 6}),
] + L((0, 50, 0, 17), (50, 70, 62, 71), (50, 70, 71, 90), (50, 70, 90, 100)))

GAP = 0.18  # the seam between joined dies


def _ultra(maxd):
    """Two Max dies joined along their short edge, the top one mirrored, as in
    Apple's M1 Ultra render."""
    H = maxd["size"][1]
    return {"dies": [(maxd, 0, 0, False, True), (maxd, 0, H + GAP, False, False)]}


def _fusion(gpu, at=0.5):
    """M5 Pro / Max: the CPU die beside the GPU die (at: where along its side,
    0 top, 1 bottom)."""
    w = M5_CPU["size"][0]
    oy = (gpu["size"][1] - M5_CPU["size"][1]) * at
    return {"dies": [(M5_CPU, 0, oy, False, False), (gpu, w + GAP, 0, False, False)]}


def _fusion_ultra():
    """M5 Ultra: two M5 Max pairs, the top pair mirrored, so both GPU dies and
    both CPU dies meet at the seam."""
    one = _fusion(M5_GPU40, at=0.0)["dies"]
    H = M5_GPU40["size"][1]
    top = [(p, ox, (H - p["size"][1] - oy), fx, True) for p, ox, oy, fx, fy in one]
    bot = [(p, ox, oy + H + GAP, fx, fy) for p, ox, oy, fx, fy in one]
    return {"dies": top + bot}


def _one(d):
    return {"dies": [(d, 0, 0, False, False)]}


PLANS = {
    "M1": _one(M1), "M1 Pro": _one(M1_PRO), "M1 Max": _one(M1_MAX), "M1 Ultra": _ultra(M1_MAX),
    "M2": _one(M2), "M2 Pro": _one(M2_PRO), "M2 Max": _one(M2_MAX), "M2 Ultra": _ultra(M2_MAX),
    "M3": _one(M3), "M3 Pro": _one(M3_PRO), "M3 Max": _one(M3_MAX), "M3 Ultra": _ultra(M3_MAX),
    "M4": _one(M4), "M4 Pro": _one(M4_PRO), "M4 Max": _one(M4_MAX),
    "M5": _one(M5), "M5 Pro": _fusion(M5_GPU20), "M5 Max": _fusion(M5_GPU40),
    "M5 Ultra": _fusion_ultra(), "M6": _one(M6),
}

# Apple's Ultra joins mirrored Max dies; 2 Ultra = 2 x Max GPU tiles.
NO_APPLE_DIE = ("M2 Ultra", "M3 Ultra", "M5 Pro", "M5 Max", "M5 Ultra", "M6")


def gpu_tiles(name):
    n = 0
    for plan, *_ in PLANS[name]["dies"]:
        n += sum(b[5]["cores"] for b in plan["blocks"] if b[0] == "gpu")
    return n


def gpu_max(name):
    """The most GPU cores the chip ships with (Apple's top config). Where the
    die shows more tiles than that (M2 Pro 20 for 19), the rest stay dark."""
    return min(gpu_tiles(name), CHIPS[name][1][0])
