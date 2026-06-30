// Centralized crop-class colour palette. Single source of truth shared by the
// map fill layer, popups, the floating legend and the chat FindingCard dots.
//
// Colours are assigned deterministically by hashing the crop label so the same
// crop always maps to the same swatch across map and chat.

/** Stable, colour-blind-aware palette keyed by hashed crop class. */
export const CROP_COLORS = [
  "#15803d", // agro green
  "#0891b2", // info cyan
  "#d97706", // amber CTA
  "#9333ea", // violet
  "#dc2626", // danger red
  "#0d9488", // teal
  "#ca8a04", // ochre
  "#2563eb", // blue
] as const;

/** Neutral grey for parcels with an unknown/missing crop class. */
export const CROP_UNKNOWN_COLOR = "#6b7280";

/** Hits/errors view colours (prediction demo): correct = green, wrong = red. */
export const CORRECT_COLOR = "#16a34a";
export const ERROR_COLOR = "#dc2626";

/** Which crop the map paints in the prediction demo. */
export type DemoView = "pred" | "truth" | "errors";

/**
 * Colour for a parcel under a given demo view: by predicted crop, by true crop,
 * or green/red by whether the prediction was correct. Used by the map fill layer
 * and the legend so both stay in sync.
 */
export function colorForDemo(
  view: DemoView,
  predCrop: string | null | undefined,
  trueCrop: string | null | undefined,
  correct: boolean | null | undefined,
): string {
  if (view === "errors") return correct ? CORRECT_COLOR : ERROR_COLOR;
  return colorForCrop(view === "truth" ? trueCrop : predCrop);
}

/** Deterministic colour for a crop label (matches legacy MapView hashing). */
export function colorForCrop(crop: string | null | undefined): string {
  if (!crop) return CROP_UNKNOWN_COLOR;
  let hash = 0;
  for (let i = 0; i < crop.length; i += 1) {
    hash = (hash * 31 + crop.charCodeAt(i)) >>> 0;
  }
  return CROP_COLORS[hash % CROP_COLORS.length] as string;
}

/** A legend entry: a crop label and its swatch colour. */
export interface CropLegendEntry {
  crop: string;
  color: string;
}

/**
 * Build a de-duplicated, sorted legend from a list of crop labels (e.g. from
 * the active findings). Null/empty labels are dropped.
 */
export function buildCropLegend(
  crops: ReadonlyArray<string | null | undefined>,
): CropLegendEntry[] {
  const seen = new Set<string>();
  const entries: CropLegendEntry[] = [];
  for (const crop of crops) {
    if (!crop || seen.has(crop)) continue;
    seen.add(crop);
    entries.push({ crop, color: colorForCrop(crop) });
  }
  entries.sort((a, b) => a.crop.localeCompare(b.crop));
  return entries;
}
