## Mermaid Flowchart Code Notes

```mermaid
graph TD;
    A[Square Rect] -- Link text --> B((Circle));
    A-->C;
    B-->D{Rhombus};
    C-->D;
    subgraph Bullet Points;
        D1[First bullet point]-->D;
        D2[Second bullet point];
        D3[Third bullet point];
    end  
```

General Arquitecture:


## Code Notes / Diagram
This is a description of how the Hyperspectral Masking algorithm is working through diagrams.


### General Arquitecture:
([Square Round])=.py
[Square Rect]=.ipynb


```mermaid


graph TD;
    CNNMasker[CNN_masker] ;
    distribution[Distribution Model];
    exploratory[exploratory_analysis];
    classifier[hyperspectral_classifier];
    masking[masking];
    MIModel[multiple_input_model]
    subgraph Source;
        distributionnet([distributionnet]);
        hyperModels([hyperspectrum_models]);
        neuralnet([neuralnet]);
        plot([plot_functions]);
    end  
```
