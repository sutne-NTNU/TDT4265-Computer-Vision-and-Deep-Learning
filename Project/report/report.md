---
figPrefix: Figure
tblPrefix: Table
numbersections: true
toc: true
date: \today
title: Project Report
subtitle: TDT4265 Computer Vision and Deep Learning
author: Sivert Utne
header-includes: |
    \fancyhead[l]{TDT4265\\\textbf{Computer Vision and Deep Learning}}
    \fancyhead[r]{Project Report\\\textbf{Sivert Utne}}
---

# General Information {-}

For convince i have created a couple scripts that can be used to reproduce each of the tasks results in the project. These are all located in the **tasks** folder. To run a task all that is needed is to (from the root of the project) do:

```sh
./tasks/<task>-<subtask>.sh
```

Due to time constraints all training was limited to 1000 iterations, with the exception of task *2.5* when training on the updated dataset with my best model.







\setcounter{section}{1}
# Model Creation

## Creating the first Baseline

The complete model is shown in [@tbl:model-baseline] and the hyperparameters used are listed in [@tbl:hyper-baseline].

\begin{table}[H]
    \centering
    \caption{The Improved Model. Using output$\_$channels $=[$128, 256, 128, 128, 64, 64$]$}
    \begin{tabular}{ | l | l | c | c | c | c | }
        \hline
        Is Output                       & Layer Type    & Number of Filters        & Kernel Size & Stride & Padding \\
        \hline
                                        &  Conv2d       &         32               &      3      &    1    &    1   \\
                                        &  ReLU         &         -                &      -      &    -    &    -   \\
                                        &  MaxPool2d    &         -                &      2      &    2    &    -   \\
                                        &  Conv2d       &         64               &      3      &    1    &    1   \\
                                        &  ReLU         &         -                &      -      &    -    &    -   \\
                                        &  Conv2d       &         64               &      3      &    1    &    1   \\
                                        &  ReLU         &         -                &      -      &    -    &    -   \\
                                        &  Conv2d       &   output$\_$channels[0]  &      3      &    2    &    1   \\
        Yes - Resolution $32\times256$  &  ReLU         &         -                &      -      &    -    &    -   \\
        \hline
                                        &  Conv2d       &         128              &      3      &    1    &    1   \\
                                        &  ReLU         &         -                &      -      &    -    &    -   \\
                                        &  Conv2d       &   output$\_$channels[1]  &      3      &    2    &    1   \\
        Yes - Resolution $16\times128$  &  ReLU         &         -                &      -      &    -    &    -   \\
        \hline
                                        &  Conv2d       &         256              &      3      &    1    &    1   \\
                                        &  ReLU         &         -                &      -      &    -    &    -   \\
                                        &  Conv2d       &   output$\_$channels[2]  &      3      &    2    &    1   \\
        Yes - Resolution $8\times64$    &  ReLU         &         -                &      -      &    -    &    -   \\
        \hline
                                        &  Conv2d       &         128              &      3      &    1    &    1   \\
                                        &  ReLU         &         -                &      -      &    -    &    -   \\
                                        &  Conv2d       &   output$\_$channels[3]  &      3      &    2    &    1   \\
        Yes - Resolution $4\times32$    &  ReLU         &         -                &      -      &    -    &    -   \\
        \hline
                                        &  Conv2d       &         128              &      3      &    1    &    1   \\
                                        &  ReLU         &         -                &      -      &    -    &    -   \\
                                        &  Conv2d       &   output$\_$channels[4]  &      3      &    2    &    1   \\
        Yes - Resolution $2\times16$    &  ReLU         &         -                &      -      &    -    &    -   \\
        \hline
                                        &  Conv2d       &         128              &      3      &    1    &    1   \\
                                        &  ReLU         &         -                &      -      &    -    &    -   \\
                                        &  Conv2d       &   output$\_$channels[5]  &      2      &    2    &    0   \\
        Yes - Resolution $1\times8$     &  ReLU         &         -                &      -      &    -    &    -   \\
        \hline
    \end{tabular}
    \label{tbl:model-baseline}
\end{table}

| Hyperparameter |  value  |
|:---------------|:-------:|
| Optimizer      |  $SGD$  |
| Batch Size     |  $32$   |
| Learning Rate  | $0.005$ |
: Hyperparameters for the improved model.
{#tbl:hyper-baseline}




## Augmenting the Data










## Implementing RetinaNet

### Feature Pyramid Network

This new model is implemented across several files. Firstly i wrapped a pretrained RetinaNet model in the file **ssd/modeling/backbones/resnet_model.py**. This model is then used as the backbone of the FPN, which is implemented in the file **ssd/modeling/backbones/fpn_model.py**. The use of this model without any further modifications are done in the config file **task_2_3_1.py**.

### Focal Loss

This change is implemented in the file *ssd/modeling/ssd_multibox_loss.py*. See config file **task_2_3_2.py** for use of these changes.

### Deep Regression and Classification Heads

This change is implemented in the file *ssd/modeling/ssd.py*. See config file **task_2_3_3.py** for use of these changes.

### Classification Head Bias

This is also implemented in the file *ssd/modeling/ssd.py*. See config file **task_2_3_4.py** for use of these changes.





## Using knowledge from the Exploration





## Extending the dataset

![*mAP* when using the model from task 2.3.4 on the extended dataset for 2500 iterations (50 epochs).](../plots/results/extended_dataset_mAP.png)

We see that my model achieves a mAP of 0.898.

\clearpage
# Discussion and Evaluation

\setcounter{subsection}{1}
<!-- ## Quantitative Analysis -->

## Discussion and Qualitative Analysis

### What are the strengths of the model?
### What are the limitations of the model?
### What is the reason for each modeling decisions impact?
### Alternative methods to the modeling decision

## Final Discussion









\clearpage
# Going Beyond

\setcounter{subsection}{1}
## Explaining the Model with CAM