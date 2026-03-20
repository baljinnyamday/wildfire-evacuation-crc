const pptxgen = require("pptxgenjs");
const path = require("path");
const React = require("react");
const ReactDOMServer = require("react-dom/server");
const sharp = require("sharp");

// Icon imports
const {
  FaFireAlt,
  FaShieldAlt,
  FaExclamationTriangle,
  FaChartBar,
  FaProjectDiagram,
  FaBrain,
  FaTable,
  FaNetworkWired,
  FaCheckCircle,
  FaCrosshairs,
  FaArrowRight,
  FaSearch,
  FaLightbulb,
  FaCode,
} = require("react-icons/fa");

// ─── Color Palette ───────────────────────────────────────────
const C = {
  bg: "111827",
  bgLight: "1F2937",
  bgCard: "374151",
  fire: "EF4444",
  fireLight: "FCA5A5",
  safe: "10B981",
  safeLight: "6EE7B7",
  amber: "F59E0B",
  amberLight: "FCD34D",
  orange: "F97316",
  white: "F9FAFB",
  gray: "9CA3AF",
  grayLight: "D1D5DB",
  grayDark: "6B7280",
  blue: "3B82F6",
  purple: "8B5CF6",
};

const FONT_H = "Georgia";
const FONT_B = "Calibri";
const FIG = path.join(__dirname, "..", "figures");

// ─── Icon Helper ─────────────────────────────────────────────
function renderIconSvg(IconComponent, color, size = 256) {
  return ReactDOMServer.renderToStaticMarkup(
    React.createElement(IconComponent, { color, size: String(size) })
  );
}

async function iconToBase64Png(IconComponent, color, size = 256) {
  const svg = renderIconSvg(IconComponent, color, size);
  const pngBuffer = await sharp(Buffer.from(svg)).png().toBuffer();
  return "image/png;base64," + pngBuffer.toString("base64");
}

// ─── Factory helpers (avoid PptxGenJS object mutation bug) ───
const makeShadow = () => ({
  type: "outer",
  color: "000000",
  blur: 8,
  offset: 3,
  angle: 135,
  opacity: 0.35,
});

const makeCardShadow = () => ({
  type: "outer",
  color: "000000",
  blur: 6,
  offset: 2,
  angle: 135,
  opacity: 0.25,
});

// ─── Main ────────────────────────────────────────────────────
async function build() {
  // Pre-render icons
  const icons = {
    fire: await iconToBase64Png(FaFireAlt, `#${C.fire}`),
    shield: await iconToBase64Png(FaShieldAlt, `#${C.safe}`),
    warning: await iconToBase64Png(FaExclamationTriangle, `#${C.amber}`),
    chart: await iconToBase64Png(FaChartBar, `#${C.blue}`),
    brain: await iconToBase64Png(FaBrain, `#${C.purple}`),
    table: await iconToBase64Png(FaTable, `#${C.orange}`),
    network: await iconToBase64Png(FaNetworkWired, `#${C.blue}`),
    check: await iconToBase64Png(FaCheckCircle, `#${C.safe}`),
    crosshair: await iconToBase64Png(FaCrosshairs, `#${C.fire}`),
    arrow: await iconToBase64Png(FaArrowRight, `#${C.white}`),
    search: await iconToBase64Png(FaSearch, `#${C.amber}`),
    lightbulb: await iconToBase64Png(FaLightbulb, `#${C.amberLight}`),
    code: await iconToBase64Png(FaCode, `#${C.grayLight}`),
    project: await iconToBase64Png(FaProjectDiagram, `#${C.purple}`),
  };

  const pres = new pptxgen();
  pres.layout = "LAYOUT_16x9";
  pres.author = "Baljinnyam Dayan";
  pres.title =
    "Conformal Risk Control for Safety-Critical Wildfire Evacuation Mapping";

  // ════════════════════════════════════════════════════════════
  // SLIDE 1: Title
  // ════════════════════════════════════════════════════════════
  {
    const s = pres.addSlide();
    s.background = { color: C.bg };

    // Subtle top accent bar
    s.addShape(pres.shapes.RECTANGLE, {
      x: 0, y: 0, w: 10, h: 0.06,
      fill: { color: C.fire },
    });

    // Fire icon
    s.addImage({ data: icons.fire, x: 4.5, y: 0.7, w: 1, h: 1 });

    // Title
    s.addText("Conformal Risk Control for\nSafety-Critical Wildfire\nEvacuation Mapping", {
      x: 0.5, y: 1.85, w: 9, h: 1.8,
      fontSize: 32, fontFace: FONT_H, color: C.white,
      bold: true, align: "center", valign: "top",
      lineSpacingMultiple: 1.15,
    });

    // Subtitle
    s.addText("A Comparative Study of Tabular, Spatial, and Graph-Based Models", {
      x: 1, y: 3.7, w: 8, h: 0.5,
      fontSize: 16, fontFace: FONT_B, color: C.gray,
      align: "center", italic: true,
    });

    // Divider line
    s.addShape(pres.shapes.LINE, {
      x: 3, y: 4.35, w: 4, h: 0,
      line: { color: C.bgCard, width: 1 },
    });

    // Author info
    s.addText("Baljinnyam Dayan", {
      x: 1, y: 4.55, w: 8, h: 0.4,
      fontSize: 18, fontFace: FONT_B, color: C.white,
      bold: true, align: "center",
    });
    s.addText("Imperial College London  |  ELEC70122  |  March 2026", {
      x: 1, y: 4.95, w: 8, h: 0.35,
      fontSize: 13, fontFace: FONT_B, color: C.grayDark,
      align: "center",
    });
  }

  // ════════════════════════════════════════════════════════════
  // SLIDE 2: Motivation — The Stakes
  // ════════════════════════════════════════════════════════════
  {
    const s = pres.addSlide();
    s.background = { color: C.bg };

    s.addText("Why This Matters", {
      x: 0.5, y: 0.3, w: 9, h: 0.6,
      fontSize: 32, fontFace: FONT_H, color: C.white, bold: true,
      margin: 0,
    });

    s.addShape(pres.shapes.RECTANGLE, {
      x: 0.5, y: 0.95, w: 1.2, h: 0.04,
      fill: { color: C.fire },
    });

    // Left column: The danger
    s.addImage({ data: icons.warning, x: 0.7, y: 1.4, w: 0.45, h: 0.45 });
    s.addText("False Negatives Kill", {
      x: 1.25, y: 1.4, w: 3.5, h: 0.45,
      fontSize: 18, fontFace: FONT_B, color: C.fire, bold: true,
      valign: "middle", margin: 0,
    });
    s.addText(
      "Evacuation perimeters are built pixel by pixel \u2014 each missed fire pixel is a gap in the safety boundary. Standard models miss 48\u201393% of fires at conventional thresholds.",
      {
        x: 0.7, y: 2.0, w: 4.1, h: 1.0,
        fontSize: 14, fontFace: FONT_B, color: C.grayLight,
        valign: "top", lineSpacingMultiple: 1.3,
      }
    );

    // Left: Shadow evacuations
    s.addImage({ data: icons.fire, x: 0.7, y: 3.2, w: 0.45, h: 0.45 });
    s.addText("Shadow Evacuations", {
      x: 1.25, y: 3.2, w: 3.5, h: 0.45,
      fontSize: 18, fontFace: FONT_B, color: C.amber, bold: true,
      valign: "middle", margin: 0,
    });
    s.addText(
      "Over-broad perimeters cause residents outside the official zone to self-evacuate, congesting roads and trapping those in genuine danger (Zhao et al., 2022).",
      {
        x: 0.7, y: 3.8, w: 4.1, h: 1.0,
        fontSize: 14, fontFace: FONT_B, color: C.grayLight,
        valign: "top", lineSpacingMultiple: 1.3,
      }
    );

    // Right: Big stat callout
    s.addShape(pres.shapes.RECTANGLE, {
      x: 5.3, y: 1.4, w: 4.2, h: 3.5,
      fill: { color: C.bgLight },
      shadow: makeShadow(),
      rectRadius: 0,
    });
    s.addText("92.8%", {
      x: 5.3, y: 1.7, w: 4.2, h: 1.2,
      fontSize: 60, fontFace: FONT_H, color: C.fire, bold: true,
      align: "center", valign: "middle",
    });
    s.addText("of fires missed by LightGBM\nat standard threshold (p \u2265 0.5)", {
      x: 5.5, y: 2.9, w: 3.8, h: 0.7,
      fontSize: 14, fontFace: FONT_B, color: C.grayLight,
      align: "center", lineSpacingMultiple: 1.3,
    });

    // Mini stats row
    s.addText("61.0%", {
      x: 5.5, y: 3.65, w: 1.8, h: 0.4,
      fontSize: 24, fontFace: FONT_H, color: C.amber, bold: true,
      align: "center",
    });
    s.addText("missed by U-Net", {
      x: 5.5, y: 4.05, w: 1.8, h: 0.3,
      fontSize: 10, fontFace: FONT_B, color: C.gray,
      align: "center",
    });
    s.addText("48.5%", {
      x: 7.5, y: 3.65, w: 1.8, h: 0.4,
      fontSize: 24, fontFace: FONT_H, color: C.amberLight, bold: true,
      align: "center",
    });
    s.addText("missed by ResGNN", {
      x: 7.5, y: 4.05, w: 1.8, h: 0.3,
      fontSize: 10, fontFace: FONT_B, color: C.gray,
      align: "center",
    });

    // Bottom takeaway
    s.addText("No model is safe at p \u2265 0.5.  We need distribution-free guarantees.", {
      x: 0.5, y: 5.0, w: 9, h: 0.4,
      fontSize: 15, fontFace: FONT_B, color: C.safe, bold: true,
      align: "center",
    });
  }

  // ════════════════════════════════════════════════════════════
  // SLIDE 3: The Gap
  // ════════════════════════════════════════════════════════════
  {
    const s = pres.addSlide();
    s.background = { color: C.bg };

    s.addText("The Research Gap", {
      x: 0.5, y: 0.3, w: 9, h: 0.6,
      fontSize: 32, fontFace: FONT_H, color: C.white, bold: true,
      margin: 0,
    });
    s.addShape(pres.shapes.RECTANGLE, {
      x: 0.5, y: 0.95, w: 1.2, h: 0.04,
      fill: { color: C.safe },
    });

    // Three columns showing the landscape
    const cols = [
      {
        icon: icons.fire, title: "Wildfire ML",
        body: "Extensive work on spread prediction (CNNs, GNNs, U-Nets). All evaluate on accuracy metrics: F1, IoU, AUROC.",
        color: C.orange, x: 0.5,
      },
      {
        icon: icons.shield, title: "Conformal Prediction",
        body: "CRC provides distribution-free safety guarantees. Applied to medical imaging, autonomous driving, NLP.",
        color: C.safe, x: 3.55,
      },
      {
        icon: icons.warning, title: "The Missing Link",
        body: "CRC covers medical imaging and autonomous driving, but wildfire spread \u2014 where false negatives are deadliest \u2014 lacks distribution-free guarantees.",
        color: C.fire, x: 6.6,
      },
    ];

    cols.forEach((c) => {
      s.addShape(pres.shapes.RECTANGLE, {
        x: c.x, y: 1.4, w: 2.85, h: 2.8,
        fill: { color: C.bgLight },
        shadow: makeCardShadow(),
      });
      s.addImage({ data: c.icon, x: c.x + 1.1, y: 1.65, w: 0.55, h: 0.55 });
      s.addText(c.title, {
        x: c.x + 0.2, y: 2.35, w: 2.45, h: 0.4,
        fontSize: 16, fontFace: FONT_B, color: c.color, bold: true,
        align: "center",
      });
      s.addText(c.body, {
        x: c.x + 0.25, y: 2.8, w: 2.35, h: 1.2,
        fontSize: 12, fontFace: FONT_B, color: C.grayLight,
        align: "center", valign: "top", lineSpacingMultiple: 1.3,
      });
    });

    // Gap statement
    s.addShape(pres.shapes.RECTANGLE, {
      x: 0.5, y: 4.55, w: 9, h: 0.75,
      fill: { color: C.bgLight },
      line: { color: C.safe, width: 1.5 },
    });
    s.addText(
      "First application of conformal risk control to wildfire spread prediction",
      {
        x: 0.7, y: 4.55, w: 8.6, h: 0.75,
        fontSize: 18, fontFace: FONT_B, color: C.safe, bold: true,
        align: "center", valign: "middle",
      }
    );
  }

  // ════════════════════════════════════════════════════════════
  // SLIDE 4: Related Work
  // ════════════════════════════════════════════════════════════
  {
    const s = pres.addSlide();
    s.background = { color: C.bg };

    s.addText("Related Work", {
      x: 0.5, y: 0.3, w: 9, h: 0.6,
      fontSize: 32, fontFace: FONT_H, color: C.white, bold: true,
      margin: 0,
    });
    s.addShape(pres.shapes.RECTANGLE, {
      x: 0.5, y: 0.95, w: 1.2, h: 0.04,
      fill: { color: C.blue },
    });

    const areas = [
      {
        icon: icons.fire, title: "Wildfire ML",
        items: [
          "Jain & Coogan (2020) \u2014 Review",
          "Radke et al. (2019) \u2014 FireCast",
          "Huot et al. (2022) \u2014 NDWS dataset",
          "All use accuracy metrics only",
        ],
        color: C.orange, x: 0.5, y: 1.3,
      },
      {
        icon: icons.shield, title: "Conformal Methods",
        items: [
          "Vovk et al. (2005) \u2014 Foundations",
          "Romano et al. (2020) \u2014 Adaptive sets",
          "Angelopoulos et al. (2022) \u2014 CRC",
          "Not applied to geospatial/disaster",
        ],
        color: C.safe, x: 5.15, y: 1.3,
      },
      {
        icon: icons.brain, title: "Calibration & UQ",
        items: [
          "Guo et al. (2017) \u2014 Miscalibration",
          "Platt (1999) \u2014 Platt scaling",
          "No finite-sample guarantees",
          "Heuristic uncertainty only",
        ],
        color: C.purple, x: 0.5, y: 3.3,
      },
      {
        icon: icons.crosshair, title: "Safety-Critical ML",
        items: [
          "Amodei et al. (2016) \u2014 AI safety",
          "FN cost \u226B FP cost in evacuation",
          "Formal guarantees needed, not just empirical perf.",
          "Wildfire: no guarantees until now",
        ],
        color: C.fire, x: 5.15, y: 3.3,
      },
    ];

    areas.forEach((a) => {
      s.addShape(pres.shapes.RECTANGLE, {
        x: a.x, y: a.y, w: 4.35, h: 1.75,
        fill: { color: C.bgLight },
        shadow: makeCardShadow(),
      });
      s.addImage({ data: a.icon, x: a.x + 0.2, y: a.y + 0.15, w: 0.35, h: 0.35 });
      s.addText(a.title, {
        x: a.x + 0.65, y: a.y + 0.15, w: 3.4, h: 0.35,
        fontSize: 15, fontFace: FONT_B, color: a.color, bold: true,
        valign: "middle", margin: 0,
      });
      s.addText(
        a.items.map((item, i) => ({
          text: item,
          options: {
            bullet: true,
            breakLine: i < a.items.length - 1,
            fontSize: 11,
            color: i === a.items.length - 1 ? a.color : C.grayLight,
            bold: i === a.items.length - 1,
          },
        })),
        {
          x: a.x + 0.25, y: a.y + 0.55, w: 3.85, h: 1.15,
          fontFace: FONT_B, valign: "top",
          paraSpaceAfter: 3,
        }
      );
    });
  }

  // ════════════════════════════════════════════════════════════
  // SLIDE 5: Our Approach Overview
  // ════════════════════════════════════════════════════════════
  {
    const s = pres.addSlide();
    s.background = { color: C.bg };

    s.addText("Our Approach", {
      x: 0.5, y: 0.3, w: 9, h: 0.6,
      fontSize: 32, fontFace: FONT_H, color: C.white, bold: true,
      margin: 0,
    });
    s.addShape(pres.shapes.RECTANGLE, {
      x: 0.5, y: 0.95, w: 1.2, h: 0.04,
      fill: { color: C.amber },
    });

    // Pipeline flow: Data -> 3 Models -> CRC -> Safe Decisions
    const steps = [
      { label: "NDWS\nDataset", sub: "18,545 samples\n64\u00d764 patches\n12 channels", icon: icons.table, color: C.blue, x: 0.3 },
      { label: "3 Model\nFamilies", sub: "LightGBM\nTiny U-Net\nResGNN-UNet", icon: icons.brain, color: C.purple, x: 2.55 },
      { label: "CRC\nCalibration", sub: "FNR \u2264 0.05\nDistribution-free\nFinite-sample", icon: icons.shield, color: C.safe, x: 4.8 },
      { label: "Safe\nDecisions", sub: "95%+ coverage\nTight zones\n3-way triage", icon: icons.check, color: C.safeLight, x: 7.05 },
    ];

    steps.forEach((st, i) => {
      s.addShape(pres.shapes.RECTANGLE, {
        x: st.x, y: 1.35, w: 2.15, h: 2.5,
        fill: { color: C.bgLight },
        shadow: makeCardShadow(),
      });
      s.addImage({ data: st.icon, x: st.x + 0.75, y: 1.55, w: 0.55, h: 0.55 });
      s.addText(st.label, {
        x: st.x + 0.1, y: 2.2, w: 1.95, h: 0.55,
        fontSize: 14, fontFace: FONT_B, color: st.color, bold: true,
        align: "center", valign: "middle",
      });
      s.addText(st.sub, {
        x: st.x + 0.1, y: 2.8, w: 1.95, h: 0.85,
        fontSize: 11, fontFace: FONT_B, color: C.grayLight,
        align: "center", valign: "top", lineSpacingMultiple: 1.3,
      });

      // Arrow between steps
      if (i < steps.length - 1) {
        s.addImage({
          data: icons.arrow,
          x: st.x + 2.15, y: 2.35, w: 0.35, h: 0.35,
        });
      }
    });

    // Key design decisions
    s.addText("Key Design Decisions", {
      x: 0.5, y: 4.15, w: 9, h: 0.4,
      fontSize: 16, fontFace: FONT_B, color: C.white, bold: true,
    });

    const decisions = [
      { text: "Deterministic splits (seed 42, 70/15/15) \u2014 calibration set never used in training", color: C.grayLight },
      { text: "Same data + splits across all models \u2014 fair architectural comparison", color: C.grayLight },
      { text: "CRC applied uniformly post-hoc \u2014 model-agnostic safety wrapper", color: C.grayLight },
    ];

    s.addText(
      decisions.map((d, i) => ({
        text: d.text,
        options: { bullet: true, breakLine: i < decisions.length - 1, fontSize: 12, color: d.color },
      })),
      {
        x: 0.7, y: 4.55, w: 8.5, h: 0.9,
        fontFace: FONT_B, paraSpaceAfter: 4,
      }
    );
  }

  // ════════════════════════════════════════════════════════════
  // SLIDE 6: What is CRC?
  // ════════════════════════════════════════════════════════════
  {
    const s = pres.addSlide();
    s.background = { color: C.bg };

    s.addText("How CRC Works", {
      x: 0.5, y: 0.3, w: 9, h: 0.6,
      fontSize: 32, fontFace: FONT_H, color: C.white, bold: true,
      margin: 0,
    });
    s.addShape(pres.shapes.RECTANGLE, {
      x: 0.5, y: 0.95, w: 1.2, h: 0.04,
      fill: { color: C.safe },
    });

    // Left: explanation
    s.addText("The Intuition", {
      x: 0.5, y: 1.3, w: 4.5, h: 0.4,
      fontSize: 18, fontFace: FONT_B, color: C.safe, bold: true,
    });

    const steps = [
      { n: "1", text: "Collect predicted probabilities on held-out calibration set" },
      { n: "2", text: "Sort the probabilities of true fire pixels" },
      { n: "3", text: 'Pick the \u03B1-quantile as threshold \u03BB\u0302 (the "safety cutoff")' },
      { n: "4", text: "Apply \u03BB\u0302 unchanged to test data \u2192 guaranteed FNR \u2264 \u03B1" },
    ];

    steps.forEach((st, i) => {
      const y = 1.85 + i * 0.65;
      s.addShape(pres.shapes.OVAL, {
        x: 0.6, y: y, w: 0.4, h: 0.4,
        fill: { color: C.safe },
      });
      s.addText(st.n, {
        x: 0.6, y: y, w: 0.4, h: 0.4,
        fontSize: 14, fontFace: FONT_B, color: C.bg, bold: true,
        align: "center", valign: "middle",
      });
      s.addText(st.text, {
        x: 1.15, y: y, w: 3.75, h: 0.4,
        fontSize: 13, fontFace: FONT_B, color: C.grayLight,
        valign: "middle",
      });
    });

    // Right: Key equation card
    s.addShape(pres.shapes.RECTANGLE, {
      x: 5.2, y: 1.3, w: 4.3, h: 1.5,
      fill: { color: C.bgLight },
      shadow: makeCardShadow(),
    });
    s.addText("The Guarantee", {
      x: 5.4, y: 1.4, w: 3.9, h: 0.35,
      fontSize: 15, fontFace: FONT_B, color: C.safe, bold: true,
      align: "center",
    });
    s.addText("E[FNR(\u03BB\u0302)] \u2264 \u03B1 = 0.05", {
      x: 5.4, y: 1.8, w: 3.9, h: 0.5,
      fontSize: 24, fontFace: "Consolas", color: C.white, bold: true,
      align: "center", valign: "middle",
    });
    s.addText("Holds regardless of model quality, calibration, or architecture", {
      x: 5.4, y: 2.35, w: 3.9, h: 0.35,
      fontSize: 11, fontFace: FONT_B, color: C.gray,
      align: "center",
    });

    // Key insight card
    s.addShape(pres.shapes.RECTANGLE, {
      x: 5.2, y: 3.1, w: 4.3, h: 1.5,
      fill: { color: C.bgLight },
      shadow: makeCardShadow(),
    });
    s.addImage({ data: icons.lightbulb, x: 5.4, y: 3.25, w: 0.35, h: 0.35 });
    s.addText("Key Insight", {
      x: 5.85, y: 3.25, w: 3.4, h: 0.35,
      fontSize: 15, fontFace: FONT_B, color: C.amberLight, bold: true,
      valign: "middle", margin: 0,
    });
    s.addText(
      "CRC decouples safety from efficiency.\nModel quality \u2192 evacuation zone size\nCRC \u2192 fire coverage guarantee",
      {
        x: 5.4, y: 3.7, w: 3.9, h: 0.8,
        fontSize: 13, fontFace: FONT_B, color: C.grayLight,
        align: "center", lineSpacingMultiple: 1.4,
      }
    );

    // FNR sweep figure
    s.addImage({
      path: path.join(FIG, "fnr_sweep.png"),
      x: 0.5, y: 4.3, w: 4.2, h: 1.15,
      sizing: { type: "contain", w: 4.2, h: 1.15 },
    });
    s.addText("FNR vs. threshold: U-Net stays safe over a much wider range", {
      x: 4.8, y: 4.7, w: 4.7, h: 0.4,
      fontSize: 11, fontFace: FONT_B, color: C.gray, italic: true,
    });
  }

  // ════════════════════════════════════════════════════════════
  // SLIDE 7: Three Architectures
  // ════════════════════════════════════════════════════════════
  {
    const s = pres.addSlide();
    s.background = { color: C.bg };

    s.addText("Three Model Families", {
      x: 0.5, y: 0.3, w: 9, h: 0.6,
      fontSize: 32, fontFace: FONT_H, color: C.white, bold: true,
      margin: 0,
    });
    s.addShape(pres.shapes.RECTANGLE, {
      x: 0.5, y: 0.95, w: 1.2, h: 0.04,
      fill: { color: C.purple },
    });

    const models = [
      {
        icon: icons.table, name: "LightGBM", type: "Tabular Baseline",
        color: C.orange, x: 0.35,
        specs: [
          "Pixel-level features (12 channels)",
          "Each pixel treated independently",
          "5% pixel subsampling for memory",
          "Binary cross-entropy objective",
        ],
        stat: "AUROC 0.854",
      },
      {
        icon: icons.brain, name: "Tiny U-Net", type: "Spatial CNN",
        color: C.safe, x: 3.4,
        specs: [
          "470K parameters",
          "2 encoder + 2 decoder blocks",
          "Skip connections, 1\u00d71 conv head",
          "AdamW + cosine annealing, 50 epochs",
        ],
        stat: "AUROC 0.969",
      },
      {
        icon: icons.network, name: "ResGNN-UNet", type: "Graph + CNN Hybrid",
        color: C.blue, x: 6.45,
        specs: [
          "229K parameters",
          "CNN encoder \u2192 GAT bottleneck \u2192 decoder",
          "3 GATConv layers, 4 heads each",
          "Early-stopped at epoch 12/25",
        ],
        stat: "AUROC 0.951",
      },
    ];

    models.forEach((m) => {
      s.addShape(pres.shapes.RECTANGLE, {
        x: m.x, y: 1.25, w: 2.9, h: 3.7,
        fill: { color: C.bgLight },
        shadow: makeCardShadow(),
      });

      // Top colored bar
      s.addShape(pres.shapes.RECTANGLE, {
        x: m.x, y: 1.25, w: 2.9, h: 0.06,
        fill: { color: m.color },
      });

      s.addImage({ data: m.icon, x: m.x + 1.1, y: 1.5, w: 0.55, h: 0.55 });
      s.addText(m.name, {
        x: m.x + 0.15, y: 2.15, w: 2.6, h: 0.35,
        fontSize: 17, fontFace: FONT_B, color: m.color, bold: true,
        align: "center",
      });
      s.addText(m.type, {
        x: m.x + 0.15, y: 2.5, w: 2.6, h: 0.3,
        fontSize: 12, fontFace: FONT_B, color: C.gray, italic: true,
        align: "center",
      });

      // Specs
      s.addText(
        m.specs.map((sp, i) => ({
          text: sp,
          options: { bullet: true, breakLine: i < m.specs.length - 1, fontSize: 11, color: C.grayLight },
        })),
        {
          x: m.x + 0.2, y: 2.9, w: 2.5, h: 1.3,
          fontFace: FONT_B, paraSpaceAfter: 4, valign: "top",
        }
      );

      // AUROC badge
      s.addShape(pres.shapes.RECTANGLE, {
        x: m.x + 0.5, y: 4.35, w: 1.9, h: 0.45,
        fill: { color: C.bgCard },
      });
      s.addText(m.stat, {
        x: m.x + 0.5, y: 4.35, w: 1.9, h: 0.45,
        fontSize: 16, fontFace: "Consolas", color: m.color, bold: true,
        align: "center", valign: "middle",
      });
    });

    // Bottom insight
    s.addText("Increasing complexity: pixel-independent \u2192 local spatial \u2192 graph propagation", {
      x: 0.5, y: 5.15, w: 9, h: 0.3,
      fontSize: 13, fontFace: FONT_B, color: C.gray, italic: true,
      align: "center",
    });
  }

  // ════════════════════════════════════════════════════════════
  // SLIDE 8: Three-Way CRC Extension
  // ════════════════════════════════════════════════════════════
  {
    const s = pres.addSlide();
    s.background = { color: C.bg };

    s.addText("Three-Way CRC Extension", {
      x: 0.5, y: 0.3, w: 9, h: 0.6,
      fontSize: 32, fontFace: FONT_H, color: C.white, bold: true,
      margin: 0,
    });
    s.addShape(pres.shapes.RECTANGLE, {
      x: 0.5, y: 0.95, w: 1.2, h: 0.04,
      fill: { color: C.amber },
    });

    // Zone cards
    const zones = [
      {
        name: "SAFE", condition: "p\u0302 < \u03BB_min", action: "No action needed",
        color: C.safe, x: 0.5,
      },
      {
        name: "MONITOR", condition: "\u03BB_min \u2264 p\u0302 < \u03BB_max", action: "Human review",
        color: C.amber, x: 3.55,
      },
      {
        name: "EVACUATE", condition: "p\u0302 \u2265 \u03BB_max", action: "Immediate evacuation",
        color: C.fire, x: 6.6,
      },
    ];

    zones.forEach((z) => {
      s.addShape(pres.shapes.RECTANGLE, {
        x: z.x, y: 1.3, w: 2.85, h: 1.6,
        fill: { color: C.bgLight },
        shadow: makeCardShadow(),
      });
      s.addShape(pres.shapes.RECTANGLE, {
        x: z.x, y: 1.3, w: 2.85, h: 0.06,
        fill: { color: z.color },
      });
      s.addText(z.name, {
        x: z.x + 0.1, y: 1.5, w: 2.65, h: 0.45,
        fontSize: 22, fontFace: FONT_H, color: z.color, bold: true,
        align: "center",
      });
      s.addText(z.condition, {
        x: z.x + 0.1, y: 1.95, w: 2.65, h: 0.35,
        fontSize: 14, fontFace: "Consolas", color: C.white,
        align: "center",
      });
      s.addText(z.action, {
        x: z.x + 0.1, y: 2.35, w: 2.65, h: 0.35,
        fontSize: 13, fontFace: FONT_B, color: C.gray, italic: true,
        align: "center",
      });
    });

    // Arrows between zones
    s.addImage({ data: icons.arrow, x: 3.35, y: 1.85, w: 0.25, h: 0.25 });
    s.addImage({ data: icons.arrow, x: 6.38, y: 1.85, w: 0.25, h: 0.25 });

    // Key parameters
    s.addText("Cost Parameters", {
      x: 0.5, y: 3.2, w: 4.5, h: 0.35,
      fontSize: 16, fontFace: FONT_B, color: C.white, bold: true,
    });

    const params = [
      "c_fn = 5 (missed fire cost) vs c_fp = 1 (false alarm cost)",
      "Asymmetric: missing fire risks lives; over-alerting causes logistics issues",
      "Shift interval \u03C1 \u2208 [0.9, 1.1] for mild prevalence variation",
    ];

    s.addText(
      params.map((p, i) => ({
        text: p,
        options: { bullet: true, breakLine: i < params.length - 1, fontSize: 12, color: C.grayLight },
      })),
      {
        x: 0.7, y: 3.6, w: 8.5, h: 0.9,
        fontFace: FONT_B, paraSpaceAfter: 4,
      }
    );

    // Key insight about MONITOR zone
    s.addShape(pres.shapes.RECTANGLE, {
      x: 0.5, y: 4.65, w: 9, h: 0.7,
      fill: { color: C.bgLight },
      line: { color: C.amber, width: 1.5 },
    });
    s.addImage({ data: icons.lightbulb, x: 0.7, y: 4.75, w: 0.4, h: 0.4 });
    s.addText(
      "Routes uncertain pixels to human review instead of forcing binary automated decisions",
      {
        x: 1.2, y: 4.65, w: 8.1, h: 0.7,
        fontSize: 15, fontFace: FONT_B, color: C.amberLight, bold: true,
        valign: "middle",
      }
    );
  }

  // ════════════════════════════════════════════════════════════
  // SLIDE 9: Why Standard Thresholds Fail (figure)
  // ════════════════════════════════════════════════════════════
  {
    const s = pres.addSlide();
    s.background = { color: C.bg };

    s.addText("Why Standard Thresholds Fail", {
      x: 0.5, y: 0.2, w: 9, h: 0.55,
      fontSize: 30, fontFace: FONT_H, color: C.white, bold: true,
      margin: 0,
    });
    s.addShape(pres.shapes.RECTANGLE, {
      x: 0.5, y: 0.78, w: 1.2, h: 0.04,
      fill: { color: C.fire },
    });

    // Figure
    s.addImage({
      path: path.join(FIG, "probability_cross_section.png"),
      x: 0.3, y: 1.0, w: 9.4, h: 3.5,
      sizing: { type: "contain", w: 9.4, h: 3.5 },
    });

    // Caption
    s.addText(
      "Probability cross-section through a U-Net prediction. Standard threshold (p \u2265 0.5, gray) misses most of the fire region. CRC threshold (\u03BB\u0302 = 0.002, green) captures it entirely.",
      {
        x: 0.5, y: 4.6, w: 9, h: 0.8,
        fontSize: 13, fontFace: FONT_B, color: C.gray, italic: true,
        align: "center", lineSpacingMultiple: 1.3,
      }
    );
  }

  // ════════════════════════════════════════════════════════════
  // SLIDE 10: Main Results
  // ════════════════════════════════════════════════════════════
  {
    const s = pres.addSlide();
    s.background = { color: C.bg };

    s.addText("Results: CRC Delivers Safety", {
      x: 0.5, y: 0.3, w: 9, h: 0.6,
      fontSize: 32, fontFace: FONT_H, color: C.white, bold: true,
      margin: 0,
    });
    s.addShape(pres.shapes.RECTANGLE, {
      x: 0.5, y: 0.95, w: 1.2, h: 0.04,
      fill: { color: C.safe },
    });

    // Results table
    const tableRows = [
      [
        { text: "Method", options: { bold: true, color: "F9FAFB", fill: { color: "374151" }, fontSize: 12, fontFace: FONT_B } },
        { text: "Params", options: { bold: true, color: "F9FAFB", fill: { color: "374151" }, fontSize: 12, fontFace: FONT_B, align: "center" } },
        { text: "\u03BB\u0302", options: { bold: true, color: "F9FAFB", fill: { color: "374151" }, fontSize: 12, fontFace: FONT_B, align: "center" } },
        { text: "Coverage \u2191", options: { bold: true, color: "F9FAFB", fill: { color: "374151" }, fontSize: 12, fontFace: FONT_B, align: "center" } },
        { text: "FNR \u2193", options: { bold: true, color: "F9FAFB", fill: { color: "374151" }, fontSize: 12, fontFace: FONT_B, align: "center" } },
        { text: "Set Size \u2193", options: { bold: true, color: "F9FAFB", fill: { color: "374151" }, fontSize: 12, fontFace: FONT_B, align: "center" } },
        { text: "AUROC \u2191", options: { bold: true, color: "F9FAFB", fill: { color: "374151" }, fontSize: 12, fontFace: FONT_B, align: "center" } },
      ],
      // LightGBM rows
      ...makeTableSection("LGBM (p\u22650.5)", "\u2014", ".500", "7.2%", ".928", "0.1%", ".854", C.bgLight, C.fire),
      ...makeTableSection("LGBM + CRC", "\u2014", ".003", "94.1%", ".059", "62.6%", ".854", C.bgLight, C.safe),
      // U-Net rows
      ...makeTableSection("U-Net (p\u22650.5)", "470K", ".500", "39.0%", ".610", "0.7%", ".969", "1F2937", C.fire),
      ...makeTableSection("U-Net + CRC", "470K", ".002", "94.7%", ".053", "14.9%", ".969", "1F2937", C.safe, true),
      // ResGNN rows
      ...makeTableSection("ResGNN (p\u22650.5)", "229K", ".500", "51.5%", ".485", "\u2014", ".951", C.bgLight, C.amber),
    ];

    s.addTable(tableRows, {
      x: 0.5, y: 1.2, w: 9, h: 2.8,
      border: { pt: 0.5, color: "4B5563" },
      colW: [2.0, 0.9, 0.8, 1.2, 0.9, 1.2, 1.0],
      rowH: [0.38, 0.36, 0.36, 0.36, 0.36, 0.36],
      autoPage: false,
    });

    // Key callout
    s.addShape(pres.shapes.RECTANGLE, {
      x: 0.5, y: 4.25, w: 4.2, h: 1.1,
      fill: { color: C.bgLight },
      shadow: makeCardShadow(),
    });
    s.addText("4.2\u00d7", {
      x: 0.5, y: 4.3, w: 1.5, h: 0.9,
      fontSize: 40, fontFace: FONT_H, color: C.safe, bold: true,
      align: "center", valign: "middle",
    });
    s.addText("smaller evacuation zones\nU-Net vs LightGBM\nunder same safety guarantee", {
      x: 2.0, y: 4.3, w: 2.5, h: 0.9,
      fontSize: 12, fontFace: FONT_B, color: C.grayLight,
      valign: "middle", lineSpacingMultiple: 1.3,
    });

    s.addShape(pres.shapes.RECTANGLE, {
      x: 5.3, y: 4.25, w: 4.2, h: 1.1,
      fill: { color: C.bgLight },
      shadow: makeCardShadow(),
    });
    s.addText("95%+", {
      x: 5.3, y: 4.3, w: 1.5, h: 0.9,
      fontSize: 40, fontFace: FONT_H, color: C.safe, bold: true,
      align: "center", valign: "middle",
    });
    s.addText("fire coverage achieved\nby both models with CRC\nregardless of architecture", {
      x: 6.8, y: 4.3, w: 2.5, h: 0.9,
      fontSize: 12, fontFace: FONT_B, color: C.grayLight,
      valign: "middle", lineSpacingMultiple: 1.3,
    });
  }

  // ════════════════════════════════════════════════════════════
  // SLIDE 11: Visual Before/After
  // ════════════════════════════════════════════════════════════
  {
    const s = pres.addSlide();
    s.background = { color: C.bg };

    s.addText("Before vs. After CRC", {
      x: 0.5, y: 0.2, w: 9, h: 0.55,
      fontSize: 30, fontFace: FONT_H, color: C.white, bold: true,
      margin: 0,
    });
    s.addShape(pres.shapes.RECTANGLE, {
      x: 0.5, y: 0.78, w: 1.2, h: 0.04,
      fill: { color: C.safe },
    });

    s.addImage({
      path: path.join(FIG, "before_after_crc.png"),
      x: 0.2, y: 1.0, w: 9.6, h: 3.3,
      sizing: { type: "contain", w: 9.6, h: 3.3 },
    });

    s.addText(
      "Three test samples (0.5%, 2.7%, 7.8% fire prevalence). Red = missed fire. Green = correctly detected. Three-way zones show SAFE/MONITOR/EVACUATE routing.",
      {
        x: 0.5, y: 4.4, w: 9, h: 0.6,
        fontSize: 12, fontFace: FONT_B, color: C.gray, italic: true,
        align: "center", lineSpacingMultiple: 1.3,
      }
    );

    // Bottom stat
    s.addText("Standard threshold: 34\u2013100% fires missed  \u2192  CRC: 0\u20132% fires missed", {
      x: 0.5, y: 5.05, w: 9, h: 0.35,
      fontSize: 15, fontFace: FONT_B, color: C.safe, bold: true,
      align: "center",
    });
  }

  // ════════════════════════════════════════════════════════════
  // SLIDE 12: The Key Insight (with three-model comparison figure)
  // ════════════════════════════════════════════════════════════
  {
    const s = pres.addSlide();
    s.background = { color: C.bg };

    s.addText("The Central Finding", {
      x: 0.5, y: 0.2, w: 9, h: 0.55,
      fontSize: 30, fontFace: FONT_H, color: C.white, bold: true,
      margin: 0,
    });
    s.addShape(pres.shapes.RECTANGLE, {
      x: 0.5, y: 0.78, w: 1.2, h: 0.04,
      fill: { color: C.amberLight },
    });

    // Statement
    s.addShape(pres.shapes.RECTANGLE, {
      x: 0.5, y: 1.05, w: 9, h: 0.65,
      fill: { color: C.bgLight },
      line: { color: C.amberLight, width: 1.5 },
    });
    s.addText(
      "Model architecture determines evacuation efficiency.  CRC determines safety.",
      {
        x: 0.7, y: 1.05, w: 8.6, h: 0.65,
        fontSize: 18, fontFace: FONT_B, color: C.amberLight, bold: true,
        align: "center", valign: "middle",
      }
    );

    // Three-model comparison figure
    s.addImage({
      path: path.join(FIG, "three_model_comparison.png"),
      x: 0.3, y: 1.9, w: 9.4, h: 2.8,
      sizing: { type: "contain", w: 9.4, h: 2.8 },
    });

    // Bottom: complexity lesson
    s.addText(
      "Complexity \u2260 better separation: ResGNN-UNet (229K, graph attention) achieves lower AUROC than Tiny U-Net (470K, standard conv) on 64\u00d764 patches.",
      {
        x: 0.5, y: 4.85, w: 9, h: 0.6,
        fontSize: 14, fontFace: FONT_B, color: C.grayLight,
        align: "center", lineSpacingMultiple: 1.3,
      }
    );
  }

  // ════════════════════════════════════════════════════════════
  // SLIDE 13: Three-Way Results & Limitation
  // ════════════════════════════════════════════════════════════
  {
    const s = pres.addSlide();
    s.background = { color: C.bg };

    s.addText("Three-Way Zone Analysis", {
      x: 0.5, y: 0.3, w: 9, h: 0.6,
      fontSize: 32, fontFace: FONT_H, color: C.white, bold: true,
      margin: 0,
    });
    s.addShape(pres.shapes.RECTANGLE, {
      x: 0.5, y: 0.95, w: 1.2, h: 0.04,
      fill: { color: C.amber },
    });

    // Zone results table
    const zoneRows = [
      [
        { text: "Model", options: { bold: true, color: "F9FAFB", fill: { color: "374151" }, fontSize: 13, fontFace: FONT_B } },
        { text: "SAFE", options: { bold: true, color: "10B981", fill: { color: "374151" }, fontSize: 13, fontFace: FONT_B, align: "center" } },
        { text: "MONITOR", options: { bold: true, color: "F59E0B", fill: { color: "374151" }, fontSize: 13, fontFace: FONT_B, align: "center" } },
        { text: "EVACUATE", options: { bold: true, color: "EF4444", fill: { color: "374151" }, fontSize: 13, fontFace: FONT_B, align: "center" } },
      ],
      [
        { text: "LightGBM", options: { color: "D1D5DB", fill: { color: "1F2937" }, fontSize: 13, fontFace: FONT_B } },
        { text: "0.0%", options: { color: "EF4444", fill: { color: "1F2937" }, fontSize: 13, fontFace: FONT_B, align: "center", bold: true } },
        { text: "95.96%", options: { color: "F59E0B", fill: { color: "1F2937" }, fontSize: 13, fontFace: FONT_B, align: "center" } },
        { text: "4.04%", options: { color: "10B981", fill: { color: "1F2937" }, fontSize: 13, fontFace: FONT_B, align: "center" } },
      ],
      [
        { text: "U-Net", options: { color: "D1D5DB", fill: { color: "111827" }, fontSize: 13, fontFace: FONT_B } },
        { text: "0.0%", options: { color: "EF4444", fill: { color: "111827" }, fontSize: 13, fontFace: FONT_B, align: "center", bold: true } },
        { text: "94.90%", options: { color: "F59E0B", fill: { color: "111827" }, fontSize: 13, fontFace: FONT_B, align: "center" } },
        { text: "5.10%", options: { color: "10B981", fill: { color: "111827" }, fontSize: 13, fontFace: FONT_B, align: "center" } },
      ],
    ];

    s.addTable(zoneRows, {
      x: 0.5, y: 1.2, w: 9, h: 1.4,
      border: { pt: 0.5, color: "4B5563" },
      colW: [2.5, 2.0, 2.5, 2.0],
      rowH: [0.42, 0.42, 0.42],
      autoPage: false,
    });

    // Explanation card
    s.addShape(pres.shapes.RECTANGLE, {
      x: 0.5, y: 2.85, w: 9, h: 2.2,
      fill: { color: C.bgLight },
      shadow: makeCardShadow(),
    });

    s.addImage({ data: icons.warning, x: 0.8, y: 3.0, w: 0.4, h: 0.4 });
    s.addText("The SAFE Zone Collapses", {
      x: 1.3, y: 3.0, w: 7.9, h: 0.4,
      fontSize: 18, fontFace: FONT_B, color: C.fire, bold: true,
      valign: "middle", margin: 0,
    });

    s.addText("With ~5% fire prevalence:", {
      x: 0.8, y: 3.55, w: 8.4, h: 0.3,
      fontSize: 14, fontFace: FONT_B, color: C.grayLight,
    });

    s.addText("B_pw = max(5 \u00d7 0.05, 1 \u00d7 0.95) = 0.95", {
      x: 1.5, y: 3.9, w: 7, h: 0.4,
      fontSize: 18, fontFace: "Consolas", color: C.amber, bold: true,
    });

    s.addText(
      "Dominated by majority-class term c_fp \u00b7 \u03C0\u2080. The shift correction exceeds the CRC threshold, pushing \u03BB_min below zero. All non-evacuated pixels routed to human review.",
      {
        x: 0.8, y: 4.35, w: 8.4, h: 0.6,
        fontSize: 13, fontFace: FONT_B, color: C.grayLight,
        lineSpacingMultiple: 1.3,
      }
    );

    // Bottom: this is a fundamental limitation
    s.addText("Fundamental limitation of prevalence-weighted bounds in rare-event regimes (\u03C0\u2080 \u226B \u03C0\u2081)", {
      x: 0.5, y: 5.15, w: 9, h: 0.3,
      fontSize: 13, fontFace: FONT_B, color: C.amber, italic: true,
      align: "center",
    });
  }

  // ════════════════════════════════════════════════════════════
  // SLIDE 14: CRC Deep Dive (figure)
  // ════════════════════════════════════════════════════════════
  {
    const s = pres.addSlide();
    s.background = { color: C.bg };

    s.addText("CRC Deep Dive: Same Model, Different Thresholds", {
      x: 0.5, y: 0.2, w: 9, h: 0.55,
      fontSize: 26, fontFace: FONT_H, color: C.white, bold: true,
      margin: 0,
    });
    s.addShape(pres.shapes.RECTANGLE, {
      x: 0.5, y: 0.78, w: 1.2, h: 0.04,
      fill: { color: C.safe },
    });

    s.addImage({
      path: path.join(FIG, "crc_deep_dive.png"),
      x: 0.2, y: 0.95, w: 9.6, h: 3.8,
      sizing: { type: "contain", w: 9.6, h: 3.8 },
    });

    s.addText(
      "Standard thresholding (p \u2265 0.5) catches 56% of fires. CRC (\u03BB\u0302 = 0.002) catches 100%. Green shading marks pixels saved by CRC.",
      {
        x: 0.5, y: 4.85, w: 9, h: 0.5,
        fontSize: 13, fontFace: FONT_B, color: C.gray, italic: true,
        align: "center", lineSpacingMultiple: 1.3,
      }
    );
  }

  // ════════════════════════════════════════════════════════════
  // SLIDE 15: Conclusion
  // ════════════════════════════════════════════════════════════
  {
    const s = pres.addSlide();
    s.background = { color: C.bg };

    s.addText("Conclusion", {
      x: 0.5, y: 0.3, w: 9, h: 0.6,
      fontSize: 32, fontFace: FONT_H, color: C.white, bold: true,
      margin: 0,
    });
    s.addShape(pres.shapes.RECTANGLE, {
      x: 0.5, y: 0.95, w: 1.2, h: 0.04,
      fill: { color: C.safe },
    });

    const findings = [
      {
        n: "1", title: "No model is safe without CRC",
        body: "Standard thresholding captures only 7\u201352% of fires across all architectures.",
        color: C.fire,
      },
      {
        n: "2", title: "CRC guarantees safety regardless of model",
        body: "Both calibrated models achieve \u226594% coverage on held-out test data.",
        color: C.safe,
      },
      {
        n: "3", title: "Model quality determines efficiency, not safety",
        body: "Tiny U-Net achieves 4.2\u00d7 tighter evacuation zones than LightGBM under identical CRC guarantee.",
        color: C.blue,
      },
      {
        n: "4", title: "Complexity without separation is waste",
        body: "Graph-augmented ResGNN-UNet underperforms standard U-Net in AUROC despite more complex architecture.",
        color: C.amber,
      },
    ];

    findings.forEach((f, i) => {
      const y = 1.25 + i * 0.95;
      s.addShape(pres.shapes.RECTANGLE, {
        x: 0.5, y, w: 9, h: 0.8,
        fill: { color: C.bgLight },
        shadow: makeCardShadow(),
      });
      // Number circle
      s.addShape(pres.shapes.OVAL, {
        x: 0.7, y: y + 0.17, w: 0.46, h: 0.46,
        fill: { color: f.color },
      });
      s.addText(f.n, {
        x: 0.7, y: y + 0.17, w: 0.46, h: 0.46,
        fontSize: 16, fontFace: FONT_B, color: C.bg, bold: true,
        align: "center", valign: "middle",
      });
      s.addText(f.title, {
        x: 1.35, y: y + 0.05, w: 7.9, h: 0.35,
        fontSize: 16, fontFace: FONT_B, color: f.color, bold: true,
        valign: "middle", margin: 0,
      });
      s.addText(f.body, {
        x: 1.35, y: y + 0.4, w: 7.9, h: 0.35,
        fontSize: 13, fontFace: FONT_B, color: C.grayLight,
        valign: "middle", margin: 0,
      });
    });

    // Practitioner message
    s.addShape(pres.shapes.RECTANGLE, {
      x: 0.5, y: 5.0, w: 9, h: 0.45,
      fill: { color: C.bgLight },
      line: { color: C.safe, width: 1 },
    });
    s.addText(
      "A simple model + CRC provides stronger safety AND tighter zones than a complex model evaluated on F1 alone.",
      {
        x: 0.7, y: 5.0, w: 8.6, h: 0.45,
        fontSize: 13, fontFace: FONT_B, color: C.safe, bold: true,
        align: "center", valign: "middle",
      }
    );
  }

  // ════════════════════════════════════════════════════════════
  // SLIDE 16: Future Work
  // ════════════════════════════════════════════════════════════
  {
    const s = pres.addSlide();
    s.background = { color: C.bg };

    s.addText("Future Work", {
      x: 0.5, y: 0.3, w: 9, h: 0.6,
      fontSize: 32, fontFace: FONT_H, color: C.white, bold: true,
      margin: 0,
    });
    s.addShape(pres.shapes.RECTANGLE, {
      x: 0.5, y: 0.95, w: 1.2, h: 0.04,
      fill: { color: C.purple },
    });

    const directions = [
      {
        icon: icons.crosshair, title: "Class-Conditional CRC",
        body: "Calibrate each zone boundary independently (FNR-CRC for \u03BB_max, FPR-CRC for \u03BB_min) to resolve SAFE zone collapse under imbalance.",
        color: C.safe,
      },
      {
        icon: icons.project, title: "Image-Level CRC",
        body: "Move from pixel-pooled to per-image FNR losses. Restores theoretical exchangeability, produces operationally meaningful per-image guarantees.",
        color: C.blue,
      },
      {
        icon: icons.chart, title: "AUROC\u2013Efficiency Formalization",
        body: "Derive closed-form relationship between model discrimination and evacuation cost. Answer: what AUROC for a target zone size?",
        color: C.amber,
      },
      {
        icon: icons.fire, title: "Temporal & Geographic Shift",
        body: "Validate CRC robustness under seasonal/climate shift using temporally stratified splits and multi-region datasets.",
        color: C.fire,
      },
    ];

    directions.forEach((d, i) => {
      const row = Math.floor(i / 2);
      const col = i % 2;
      const x = 0.5 + col * 4.65;
      const y = 1.25 + row * 1.85;

      s.addShape(pres.shapes.RECTANGLE, {
        x, y, w: 4.35, h: 1.6,
        fill: { color: C.bgLight },
        shadow: makeCardShadow(),
      });
      s.addImage({ data: d.icon, x: x + 0.2, y: y + 0.15, w: 0.4, h: 0.4 });
      s.addText(d.title, {
        x: x + 0.7, y: y + 0.15, w: 3.4, h: 0.4,
        fontSize: 15, fontFace: FONT_B, color: d.color, bold: true,
        valign: "middle", margin: 0,
      });
      s.addText(d.body, {
        x: x + 0.2, y: y + 0.6, w: 3.95, h: 0.9,
        fontSize: 11.5, fontFace: FONT_B, color: C.grayLight,
        valign: "top", lineSpacingMultiple: 1.3,
      });
    });
  }

  // ════════════════════════════════════════════════════════════
  // SLIDE 17: Thank You
  // ════════════════════════════════════════════════════════════
  {
    const s = pres.addSlide();
    s.background = { color: C.bg };

    // Top accent bar
    s.addShape(pres.shapes.RECTANGLE, {
      x: 0, y: 0, w: 10, h: 0.06,
      fill: { color: C.safe },
    });

    s.addImage({ data: icons.shield, x: 4.5, y: 1.0, w: 1, h: 1 });

    s.addText("Thank You", {
      x: 1, y: 2.2, w: 8, h: 0.8,
      fontSize: 40, fontFace: FONT_H, color: C.white, bold: true,
      align: "center",
    });

    s.addText("Questions?", {
      x: 1, y: 3.0, w: 8, h: 0.5,
      fontSize: 22, fontFace: FONT_B, color: C.safe,
      align: "center",
    });

    // Contact info
    s.addShape(pres.shapes.LINE, {
      x: 3.5, y: 3.7, w: 3, h: 0,
      line: { color: C.bgCard, width: 1 },
    });

    s.addText("Baljinnyam Dayan", {
      x: 1, y: 3.9, w: 8, h: 0.35,
      fontSize: 16, fontFace: FONT_B, color: C.white, bold: true,
      align: "center",
    });
    s.addText("baljinnyam.dayan25@imperial.ac.uk", {
      x: 1, y: 4.25, w: 8, h: 0.3,
      fontSize: 13, fontFace: FONT_B, color: C.gray,
      align: "center",
    });

    s.addImage({ data: icons.code, x: 2.3, y: 4.75, w: 0.3, h: 0.3 });
    s.addText("github.com/baljinnyamday/wildfire-evacuation-crc", {
      x: 2.65, y: 4.75, w: 5.5, h: 0.3,
      fontSize: 12, fontFace: FONT_B, color: C.grayDark,
      valign: "middle",
    });
  }

  // ─── Write ─────────────────────────────────────────────────
  await pres.writeFile({ fileName: path.join(__dirname, "presentation.pptx") });
  console.log("Done: presentation/presentation.pptx");
}

// ─── Table row helper ────────────────────────────────────────
function makeTableSection(method, params, lambda, cov, fnr, setSize, auroc, bgColor, covColor, highlight = false) {
  const opts = (text, color = "D1D5DB", align = "center", bold = false) => ({
    text,
    options: {
      color,
      fill: { color: bgColor },
      fontSize: 12,
      fontFace: "Calibri",
      align,
      bold,
    },
  });

  return [
    [
      opts(method, "D1D5DB", "left", highlight),
      opts(params),
      opts(lambda),
      opts(cov, covColor, "center", true),
      opts(fnr, covColor === "EF4444" ? "FCA5A5" : "6EE7B7"),
      opts(setSize),
      opts(auroc),
    ],
  ];
}

build().catch(console.error);
