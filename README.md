# PeKo: Precondition Knowledge Dataset

## Overview
PeKo (**P**r**e**condition **K**n**o**wledge) is a large scale crowdsourced event precondition knowledge dataset introduced in our paper _Modeling Preconditions in Text with a Crowd-sourced Dataset_ at EMNLP Findings 2020 

Preprint coming soon.

## Crowdsourcing Precondition Knowledge
We extract events and their temporal relations from news articles using CAEVO ([Chambers et al., 2014](https://www.usna.edu/Users/cs/nchamber/caevo/)), a temporal relation extraction system. We used CAEVO on a random sample of 6,837 articles inthe New York Times Annotated Corpus ([Sandhaus, 2008](https://catalog.ldc.upenn.edu/LDC2008T19)). 
On average CAEVO extracted around 63 events per article, which yielded a total of 3,906 possible relation candidates per document. We filtered these to retain only pairs of events that have a BEFORE or AFTER temporal relation between them. We call the temporally preceding event the _candidate precondition_, and the temporally subsequent event in the pair the _target event_.
![Crowdsourcing Task](images/crowdsourcing.svg)
The annotators were presented with a text snippet and two event mentions highlighted as shown below. To prune out event extraction errors from CAEVO, the annotators were first asked if the highlighted text denoted valid events. If both triggers were deemed valid, then the annotators evaluated whether or not the candidate precondition event was an actual precondition for the target event. Specifically they check if the candidate event is necessary for the target event to happen.
<p align="center">
  <img align="middle" src="images/mturk_example.png" alt="HIT example" width="400"/>
</p>

## Tasks
We now propose two tasks that test for the ability to recognize and generate preconditions in textual contexts. Here we describe evaluations to benchmark the performance of current models on these tasks and to better understand the challenges involved.

### PeKo Task 1: Precondition Identification
Given a text snippet with a target and candidate event pair, the task is to classify if the candidate event is a precondition for the target in the context described by the text snippet. This is a standard sentence-level classification task.
<p align="center">
  <img align="middle" src="images/result_table.png" alt="Result Table" width="400"/>
</p>

### PeKo Task 2: Precondition Generation Task
Here we introduce Precondition Generation as a more general challenge that a dataset like PeKo now enables. Given a target event `t`, generate an event `p` that is a precondition for `t`. We benchmark performance on evaluation instances drawn from both PeKo and an out-of-domain dataset [ATOMIC](https://homes.cs.washington.edu/~msap/atomic/).

## Download
The dataset can be downloaded from [here](https://github.com/StonyBrookNLP/PeKo)


```markdown
Syntax highlighted code block

# Header 1
## Header 2
### Header 3

- Bulleted
- List

1. Numbered
2. List

**Bold** and _Italic_ and `Code` text

[Link](url) and ![Image](src)
```

For more details see [GitHub Flavored Markdown](https://guides.github.com/features/mastering-markdown/).

### Jekyll Themes

Your Pages site will use the layout and styles from the Jekyll theme you have selected in your [repository settings](https://github.com/StonyBrookNLP/PeKo/settings). The name of this theme is saved in the Jekyll `_config.yml` configuration file.

### Support or Contact

Having trouble with Pages? Check out our [documentation](https://docs.github.com/categories/github-pages-basics/) or [contact support](https://github.com/contact) and we’ll help you sort it out.
