import {
  afterEach,
  beforeEach,
  describe,
  expect,
  it,
  vi,
  type Mock,
} from "vitest";
import { createApp, ref, watch } from "vue";
import { createPinia, setActivePinia } from "pinia";
import { useMapStore } from "~/stores/map";
import { useChatStore } from "~/stores/chat";
import { useAoi } from "~/composables/useAoi";
import { demoFindings } from "~/utils/demoPreview";

// Exercises the REAL `useMap` composable (US-058 fe/A + fe/B) under jsdom with a
// MOCKED maplibre-gl: the real map engine never loads in jsdom, so we replace
// the dynamic `import("maplibre-gl")` with a fake `Map`/`Popup`/control set that
// records `addSource`/`addLayer`/`on`/`remove` calls and lets us fire handlers.
//
// What is REAL: the composable code path, both Pinia stores, and the parcel_id —
// it is the actual id of a rendered demo feature (`demoFindings()` parcel 101),
// NOT an invented number (REGLA ARTHUR).
//
// What is mocked: maplibre-gl (no WebGL in jsdom) and the Nuxt auto-imports
// `ref`/`watch`/`useI18n` that `useMap` reads as globals at runtime.

// --- Mock maplibre-gl -------------------------------------------------------
// A captured handler registry keyed by `event` or `event:layerId` so a test can
// fire the exact listener `useMap` wired (e.g. "load", "click:findings-fill").

type Handler = (...args: unknown[]) => void;

class FakeGeoJSONSource {
  setData = vi.fn();
}

interface FakeMapState {
  handlers: Map<string, Handler[]>;
  sources: string[];
  layers: string[];
  removed: boolean;
}

let lastMap: FakeMapInstance | null = null;

class FakeMapInstance {
  state: FakeMapState = {
    handlers: new Map(),
    sources: [],
    layers: [],
    removed: false,
  };

  addControl = vi.fn();
  addSource = vi.fn((id: string) => {
    this.state.sources.push(id);
  });
  addLayer = vi.fn((spec: { id: string }) => {
    this.state.layers.push(spec.id);
  });
  getSource = vi.fn(() => new FakeGeoJSONSource());
  getLayer = vi.fn((id: string) => (this.state.layers.includes(id) ? { id } : undefined));
  setLayoutProperty = vi.fn();
  setFilter = vi.fn();
  setStyle = vi.fn();
  fitBounds = vi.fn();
  getBounds = vi.fn(() => ({
    toArray: () => [
      [11.0, 43.2],
      [11.2, 43.4],
    ],
  }));
  getCanvas = vi.fn(() => ({ style: { cursor: "" } }));
  dragPan = { enable: vi.fn(), disable: vi.fn() };
  remove = vi.fn(() => {
    this.state.removed = true;
  });

  on = vi.fn((event: string, layerOrHandler: unknown, maybeHandler?: unknown) => {
    const hasLayer = typeof layerOrHandler === "string";
    const key = hasLayer ? `${event}:${layerOrHandler as string}` : event;
    const handler = (hasLayer ? maybeHandler : layerOrHandler) as Handler;
    const list = this.state.handlers.get(key) ?? [];
    list.push(handler);
    this.state.handlers.set(key, list);
  });
  once = vi.fn();

  /** Fire every handler registered for `key`, passing `evt`. */
  fire(key: string, evt: unknown) {
    for (const h of this.state.handlers.get(key) ?? []) h(evt);
  }
}

class FakePopup {
  setLngLat = vi.fn(() => this);
  setHTML = vi.fn(() => this);
  addTo = vi.fn(() => this);
}

class FakeLngLatBounds {
  extend = vi.fn();
}

vi.mock("maplibre-gl", () => {
  // `new maplibre.Map(...)` -> a constructor that records the latest instance.
  const MapMock = vi.fn(function MapCtor() {
    lastMap = new FakeMapInstance();
    return lastMap;
  });
  return {
    Map: MapMock,
    Popup: FakePopup,
    NavigationControl: vi.fn(),
    ScaleControl: vi.fn(),
    LngLatBounds: FakeLngLatBounds,
    default: {
      Map: MapMock,
      Popup: FakePopup,
      NavigationControl: vi.fn(),
      ScaleControl: vi.fn(),
      LngLatBounds: FakeLngLatBounds,
    },
  };
});

// The dynamic `import("maplibre-gl/dist/maplibre-gl.css")` must resolve.
vi.mock("maplibre-gl/dist/maplibre-gl.css", () => ({}));

import { useMap } from "~/composables/useMap";

let app: ReturnType<typeof createApp>;

beforeEach(() => {
  app = createApp({ render: () => null });
  const pinia = createPinia();
  app.use(pinia);
  setActivePinia(pinia);

  // `useMap` reads `ref`/`watch`/`useI18n`/`useAoi` as Nuxt auto-imports
  // (globals). `useAoi` is the REAL composable (it only touches the map store),
  // so the draw path stays genuine; the rest are thin shims.
  vi.stubGlobal("ref", ref);
  vi.stubGlobal("watch", watch);
  vi.stubGlobal("useI18n", () => ({ t: (k: string) => k }));
  vi.stubGlobal("useAoi", useAoi);

  lastMap = null;
});

afterEach(() => {
  vi.unstubAllGlobals();
  vi.restoreAllMocks();
});

/** Init a map and run its "load" handler so overlays/interactions wire up. */
async function initLoadedMap(opts = {}) {
  const handle = useMap(opts);
  const container = document.createElement("div");
  await handle.initMap(container);
  // The map engine fires "load" once the style is ready; do it deterministically.
  lastMap!.fire("load", {});
  return handle;
}

describe("useMap lifecycle (mocked maplibre-gl)", () => {
  it("initMap builds the map and wires sources + layers on load", async () => {
    const handle = await initLoadedMap();

    expect(lastMap).not.toBeNull();
    // Both overlay sources added with stable ids.
    expect(lastMap!.addSource).toHaveBeenCalled();
    expect(lastMap!.state.sources).toEqual(
      expect.arrayContaining(["findings", "active-aoi"]),
    );
    // The findings fill/line/highlight + AOI fill/line layers added.
    expect(lastMap!.state.layers).toEqual(
      expect.arrayContaining([
        "findings-fill",
        "findings-line",
        "findings-highlight",
        "active-aoi-fill",
        "active-aoi-line",
      ]),
    );
    // Click + move listeners registered (the real interaction wiring).
    expect(lastMap!.on).toHaveBeenCalled();
    expect(lastMap!.state.handlers.has("click:findings-fill")).toBe(true);
    expect(lastMap!.state.handlers.has("moveend")).toBe(true);
    expect(handle.isReady.value).toBe(true);
  });

  it("initMap is idempotent (a second call does not build a second map)", async () => {
    const handle = useMap();
    const container = document.createElement("div");
    await handle.initMap(container);
    const first = lastMap;
    await handle.initMap(container);
    expect(lastMap).toBe(first);
  });

  it("destroyMap removes the map and resets ready state", async () => {
    const handle = await initLoadedMap();
    expect(handle.isReady.value).toBe(true);

    handle.destroyMap();

    expect(lastMap!.remove).toHaveBeenCalledTimes(1);
    expect(lastMap!.state.removed).toBe(true);
    expect(handle.isReady.value).toBe(false);
    expect(handle.drawRect.value).toBeNull();
  });

  it("moveend reports the visible bbox to the map store as a flat [w,s,e,n]", async () => {
    await initLoadedMap();
    const mapStore = useMapStore();

    lastMap!.fire("moveend", {});

    // getBounds() -> [[11.0,43.2],[11.2,43.4]] flattened.
    expect(mapStore.visibleBbox).toEqual([11.0, 43.2, 11.2, 43.4]);
  });
});

describe("useMap parcel click -> real parcel_id extraction", () => {
  it("extracts the REAL parcel_id from the clicked feature and calls onParcelSelect", async () => {
    const onParcelSelect: Mock = vi.fn();
    await initLoadedMap({ onParcelSelect });

    // Use a REAL rendered demo feature's properties (parcel 101, Vineyard).
    const real = demoFindings()[0]!;
    const clickEvent = {
      lngLat: { lng: 11.102, lat: 43.302 },
      features: [
        {
          properties: {
            parcel_id: real.parcel_id,
            crop_class: real.crop_class,
            confidence: real.confidence,
            ndvi_mean: real.ndvi_mean,
            area_ha: real.area_ha,
            source: real.citation.source,
          },
        },
      ],
    };

    lastMap!.fire("click:findings-fill", clickEvent);

    expect(onParcelSelect).toHaveBeenCalledTimes(1);
    const [parcelId, props] = onParcelSelect.mock.calls[0] as [
      number,
      Record<string, unknown>,
    ];
    expect(parcelId).toBe(real.parcel_id); // 101, the real feature id
    expect(props.crop_class).toBe("Vineyard");
  });

  it("does not call onParcelSelect when the feature has no parcel_id", async () => {
    const onParcelSelect: Mock = vi.fn();
    await initLoadedMap({ onParcelSelect });

    lastMap!.fire("click:findings-fill", {
      lngLat: { lng: 0, lat: 0 },
      features: [{ properties: {} }],
    });

    expect(onParcelSelect).not.toHaveBeenCalled();
  });
});

describe("useMap cross-store link: parcel click -> map store + chat store", () => {
  it("a click selects the parcel in the map store AND sets the chat store parcel_id (real id)", async () => {
    const mapStore = useMapStore();
    const chatStore = useChatStore();

    // Wire the cross-store link exactly as MapCanvas.vue does (fe/B): useMap stays
    // agnostic; the caller fans the real parcel_id into BOTH stores.
    await initLoadedMap({
      onParcelSelect: (id: number, props: Record<string, unknown>) => {
        mapStore.setSelectedParcel({
          parcel_id: id,
          crop_class: (props.crop_class as string | null) ?? null,
        });
        chatStore.setActiveParcelId(id);
      },
    });

    const real = demoFindings()[1]!; // parcel 102, Olive grove
    lastMap!.fire("click:findings-fill", {
      lngLat: { lng: 11.106, lat: 43.303 },
      features: [
        {
          properties: {
            parcel_id: real.parcel_id,
            crop_class: real.crop_class,
          },
        },
      ],
    });

    // Map store holds the highlight/chip with the real id + crop.
    expect(mapStore.hasSelectedParcel).toBe(true);
    expect(mapStore.selectedParcel).toEqual({
      parcel_id: 102,
      crop_class: "Olive grove",
    });
    // Chat store holds the id that travels to the backend on the next POST /chat.
    expect(chatStore.activeParcelId).toBe(102);
    expect(chatStore.activeParcelId).toBe(real.parcel_id);
  });

  it("clearing the selection wipes both stores in lockstep", async () => {
    const mapStore = useMapStore();
    const chatStore = useChatStore();

    await initLoadedMap({
      onParcelSelect: (id: number) => {
        mapStore.setSelectedParcel({ parcel_id: id });
        chatStore.setActiveParcelId(id);
      },
    });

    const real = demoFindings()[2]!; // parcel 103, Wheat
    lastMap!.fire("click:findings-fill", {
      lngLat: { lng: 11.103, lat: 43.307 },
      features: [{ properties: { parcel_id: real.parcel_id } }],
    });
    expect(chatStore.activeParcelId).toBe(103);

    // The chip's "clear" wires to both store clears (MapCanvas.clearParcelSelection).
    mapStore.clearSelectedParcel();
    chatStore.setActiveParcelId(null);

    expect(mapStore.hasSelectedParcel).toBe(false);
    expect(mapStore.selectedParcel).toBeNull();
    expect(chatStore.activeParcelId).toBeNull();
  });
});
