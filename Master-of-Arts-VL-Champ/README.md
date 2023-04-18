# ML Art challenge

This challenge is aimed on creation of AI that capable to determine a genre of an artwork based on its visual appearance. Visual appearance of an artwork is represented by photos that are carefully distributed into categories.

## Data

The data given in the competition has hierarchical struscture. There are 6 top level categories:
* Architecture
* Graphics
* Installations
* Sculptures
* Paintings
* Decorativa and Applied Art

Each of these top-level categories variable amount of sub-categories. For example, Paintings category contains 10 genres which are titled as Battle, Animalistic, Abstract, and etc. 

## Problem Statement

As a solution it is expected that you'll provide an algorithm that based on the image of an artwork can predict its category. The label is counted as properly determined only if both top-level category and sub-category predicted right. The predictions should be of quality and the quality wiil be measured using the metric described below. The winners of competition will be decided based on the metric of their models.

## Evaluation

Submissions are evaluated using the macro F1 score.

F1 is calculated as follows:

$$ F_1 = \frac{2 * precision * recall}{precission + recall}, $$
where 
$$ precision = \frac{TP}{TP + FP}, \, recall = \frac{TP}{TP + FN} $$

In "macro" F1 a separate F1 score is calculated for each sub-category value and then averaged.

### Submission Format
For each image Id, you should predict the corresponding image label ("category_id") in the Predicted column. The submission file should have the following format:

```
Id,Predicted
3,0
1,27
2,42
...
```