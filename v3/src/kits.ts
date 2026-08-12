export type ModelSlot = {
  slot: string;
  modelId: string;
};

export type DomainKit = {
  kitId: string;
  kitVersion: string;
  label: string;
  labelKo: string;
  workspacePages: string[];
  modelRecipe: ModelSlot[];
};

/** Static fixtures — concept preview only, not live inventory. */
export const DEMO_KITS: DomainKit[] = [
  {
    kitId: "fashion_kr_retail",
    kitVersion: "2026-08-05.1",
    label: "Fashion KR Retail",
    labelKo: "패션 리테일 (KR)",
    workspacePages: ["monitor", "collect", "drift", "ontology", "review", "models"],
    modelRecipe: [
      { slot: "image_encoder", modelId: "Marqo/marqo-fashionSigLIP" },
      { slot: "text_encoder", modelId: "intfloat/multilingual-e5-small" },
      { slot: "entity_extractor", modelId: "urchade/gliner_multi-v2.1" },
    ],
  },
  {
    kitId: "plate_quality_scaffold",
    kitVersion: "2026-08-05.1",
    label: "Vehicle Plate Recognition",
    labelKo: "차량 번호판 식별",
    workspacePages: ["drift", "ontology", "models", "review"],
    modelRecipe: [
      { slot: "car_detector", modelId: "YOLO car detector" },
      { slot: "plate_detector", modelId: "plate detector" },
      { slot: "ocr", modelId: "CRNN-CTC" },
    ],
  },
];

/** Demo “installed” flags so the linking concept is visible. */
export const DEMO_INSTALLED: Record<string, boolean> = {
  "Marqo/marqo-fashionSigLIP": true,
  "intfloat/multilingual-e5-small": true,
  "urchade/gliner_multi-v2.1": true,
  "YOLO car detector": false,
  "plate detector": false,
  "CRNN-CTC": false,
};
