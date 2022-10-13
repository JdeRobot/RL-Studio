---
title: "Speed-up inference with openVINO"
excerpt: "Speed-up inference with openVINO"

sidebar:
  nav: "docs"

toc: true
toc_label: "TOC installation"
toc_icon: "cog"


categories:
- inference openvino
tags:
- tag 1
- tag 2
- tag 3
- tag 4

author: Roberto Calvo Palomino
pinned: false
---

<strong>openVINO</strong>

OpenVINO™ is an open-source toolkit for optimizing and deploying AI inference.  
- Boost deep learning performance in computer vision, automatic speech recognition, natural language processing and other common tasks
- Use models trained with popular frameworks like TensorFlow, PyTorch and more
- Reduce resource demands and efficiently deploy on a range of Intel® platforms from edge to cloud

Check more info about the project in the following links:
- [https://github.com/openvinotoolkit/openvino](https://github.com/openvinotoolkit/openvino)
- [https://docs.openvino.ai/nightly/openvino_docs_MO_DG_Deep_Learning_Model_Optimizer_DevGuide.html](https://docs.openvino.ai/nightly/openvino_docs_MO_DG_Deep_Learning_Model_Optimizer_DevGuide.html)

AWS deepracer uses OpenVINO: http://wiki.ros.org/aws_deepracer

By using openVINO you can really speed-up the inference process even using only the CPU. It does optimize the model for specific platform.

![Flow](https://docs.openvino.ai/nightly/_images/BASIC_FLOW_MO_simplified.svg)


<strong>DEMO</strong>

For a very simple model as the one below, inference time in GPU and CPU is around 55 ms. (tested in GeForce GTX 1080, latop, CPU and Google Coral). Meaning that we can get 18-20 iterations per second at most.

```python
model = Sequential()
model.add(Dense(24, input_shape=(14,), activation="relu"))
model.add(Dense(24, activation="relu"))
model.add(Dense(7, activation="linear"))
model.compile(loss='mse', optimizer=Adam(lr=0.001))
```

Using OpenVINO optimizer and engine we can reduce the inference time to 0.25 ms! Theorically we could run that model at 4000 iter/sec. Of course, we have more actors and components in our system (image processing, control systems, ...). Anyways in the following video you can check how the control system can iterate at 90 FPS in a simple follow line model based on reinforcement learning. Awesome!

<iframe width="560" height="315" src="https://www.youtube.com/embed/RfWqEcayTJA" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
