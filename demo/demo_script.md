# Two Minute Demo

Before presenting, prepare all assets, open the app, and run a four-model comparison once. Keep `demo/example_queries/` open for manual uploads. Save screenshots of the key results as a fallback if the presentation machine becomes unavailable.

| Time | Action | Point to explain |
|---|---|---|
| 0:00–0:25 | Upload `Abyssinian_224.jpg`, select DINOv2, and search. | An uploaded image is embedded and matched against the fixed gallery. The Top-5 is computed live. |
| 0:25–1:15 | Select **Example**, choose **Where DINOv2 helps**, switch to **Compare 4 models**, and search. Use **View results** or **Next** to move between models. | The same query and gallery produce different neighbors depending on the encoder. DINOv2 finds 5/5 of the correct breed; DINO-S 0/5, DeiT-S 1/5, DeiT-B 0/5. |
| 1:15–1:40 | Choose **A shared failure** and search. | All models fail on this Siamese query; DINOv2 retrieves Birman cats. Breed agreement is an imperfect proxy for visual similarity. |
| 1:40–2:00 | Return to **Where DINOv2 helps**. Use the comparison table and browse the DINO-S and DINOv2 result views. | Contrast retrieval quality and live embedding time. Refer to the aggregate test-set results already presented in the slides. |

Reserve example for Q&A: **Larger is not always better**, `leonberger_73.jpg`. Correct Top-5 counts are DINO-S 4, DeiT-S 5, DINOv2 5, and DeiT-B 0.

Explain that no training occurs during the demo. Gallery vectors were computed beforehand; uploaded query embeddings and Top-5 search are computed on demand. Similarity is not a confidence probability. The historical T4 benchmark and live local timings use different measurement contexts.
