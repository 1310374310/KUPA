the implement of KUPA in pytorch





# Environment Requirement
The code has been tested running under Python 3.6.5. The required packages are as follows:

- pytorch == 1.6.0
- numpy == 1.19.5
- scipy == 1.5.4
- sklearn == 0.20.0
- torch_scatter == 2.0.6
- networkx == 2.5


# Reproducibility & Example to Run the Codes


- Last-FM dataset

```
python train.py  --dataset music
```

- Amazon-book dataset

```
python train.py --dataset Amazon-Books
```

- MindReader dataset

```
python train.py --dataset MindReader
```

- MovieLens dataset

```
python train.py --dataset ml-1m
```


- Alibaba-iFashion dataset

```
python train.py --dataset Alibaba-iFashion
```


# Dataset
We provide five processed datasets: Last-FM, Amazon-Books, MoveLens, MindReader, and Alibaba-iFashion.
- You can find the full version of recommendation datasets via them.
- We follow [KB4Rec](https://github.com/RUCDM/KB4Rec) to preprocess Amazon-book and MovieLens datasets, mapping items into Freebase entities via title matching if there is a mapping available.

- 'dataset.inter'
    - Interaction file.
    - Each line is a user with her/his positive interaction with an item.
- 'dataset.kg'
    - Knowledge Graph file.
    - Each line is a triplet of KG, e.g. (head_entity, relation, tail_entity). 
- 'dataset.link'
    - A map for item to entity
    - Each line is the item_id and its corresponding entity_id