# Fine-tuning effect on the sample property

`microsoft/Florence-2-base-ft` before and after 8 epochs on the five photos in
`data/sample_property/` (`data/train/sample_property.jsonl`, frozen vision tower,
lr 1e-5, ~1 minute on CPU; training loss 4.94 -> 2.02).

Five examples only overfits by design — this is a smoke test of the training loop,
not a production model. Run `build_training_data.py` over the full scraped corpus
for a real run.

| Photo | Base `Florence-2-base-ft` | Fine-tuned |
| --- | --- | --- |
| `01_exterior.jpg` | This is an image of a building. The building is red brick. There is a palm tree in front of the building. There are large green trees surrounding the house. The sky above the building is blue. The windows on the building are covered with white curtains. The ground under the house is covered in green grass. | A red brick apartment block is surrounded by lush green trees. The apartment block has four levels and plenty of windows. |
| `02_living.jpg` | The living room is empty. The couch is gray. There are four pillows on the couch. The pillows are colorful. There is a red rug on the floor. The rug has white letters on it. The floor is beige. The walls in the room are white. The windows are open. | The living and dining area of a modern apartment. The flooring is neutral and the walls are painted white. There is contemporary furniture, including a sofa, coffee table, and end table, as well as a kitchenette. |
| `03_kitchen.jpg` | The kitchen is clean. The cabinets are white. There is a dishwasher in the kitchen. The dishwasher is silver. The stove is black. There's a window in the wall. The floor is made of wood. The wood is a light brown color. | This photo is taken inside of a kitchen. White cabinetry line the walls of the kitchen. A stainless steel oven, gas cooktop, range hood and dishwasher are built into the cabinets. A double sink sits in the center of the countertop, with a window above it. |
| `04_bedroom.jpg` | A colorful bed is in a room. There is a large window on the wall next to the bed. There are colorful pillows on top of the bed with a colorful blanket on it. | Sunlight shines through a bedroom with a queen bed and bedside furniture. The bedspread and pillows are brightly colored and the carpet is a neutral colour. |
| `05_bathroom.jpg` | A white toilet sits in the middle of a bathroom. There is a window above the toilet. There are two sinks in front of the mirror. | Bathroom with a separate bathtub, separate toilet and separate vanity with a large mirror. The walls are tiled in shades of green, and the floor is tiled with small square tiles. |

The vocabulary moves from generic captioning ("the stove is black") to listing terms
("stainless steel oven, gas cooktop, range hood and dishwasher"). It also inherits the
failure mode of a tiny training set: the exterior caption invents "four levels", and the
living room picks up "kitchenette" from a neighbouring example. Both are hallucinations —
any production use needs the review step described in the README.
