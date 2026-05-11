export default class CardSelector {
  constructor(registry = []) {
    this.registry = registry;
  }

  selectCards({ artifactIndex, payloads }) {
    if (!artifactIndex || !Array.isArray(artifactIndex.artifacts)) return [];
    const cards = [];

    artifactIndex.artifacts.forEach((artifact) => {
      const payload =
        payloads?.[artifact.id] ??
        (artifact.payload?.mode === "inline" ? artifact.payload.data : null);

      this.registry.forEach((card) => {
        if (
          card.supportedArtifactTypes &&
          !card.supportedArtifactTypes.includes(artifact.type)
        ) {
          return;
        }
        if (card.match && !card.match({ artifact, payload, artifactIndex, payloads })) {
          return;
        }
        cards.push({
          ...card,
          artifact,
          data: card.extractData
            ? card.extractData({ artifact, payload, artifactIndex, payloads })
            : payload,
        });
      });
    });

    return cards.sort((a, b) => (a.priority || 0) - (b.priority || 0));
  }
}
