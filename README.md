# AI Explanatory Drivers - VS Past Pipeline

## Methodology

![VS Past Methodology](src/VS_past_methodology.png)

The AI Explanatory Drivers tool explains the NPS, which is computed as the difference in probabilities of being a promoter and being a detractor within a sample of clients, based on their response to a generic satisfaction question (nps_100).

The pipeline contains two binary classifier models, both operating on a client level:
- One model outputs the probability of being a promoter.
- The other model outputs the probability of being a detractor given a certain survey, excluding the generic question.

These probabilities are used to compute the NPS per client as their difference in probabilities. The NPS of a sample of clients is then predicted as the mean client NPS within the sample.

The key feature of this procedure is computing the NPS without relying on the generic question of the survey. Instead, it uses the rest of the questions and some external variables to provide insights into what has driven changes in the NPS over different periods of time, hence the name AI Explanatory Drivers.

This part of the tool is named the "VS Past" pipeline because it operates on a client level, requiring the exact population that generated a certain NPS, which is only accessible for past periods of time.

## Explainability with Shapley Values

The tool uses Shapley values to explain each prediction for both models. The Shapley values output values in the logistic space, so an inv_logistic transformation is applied to convert them into the probabilistic space. Although this is not mathematically exact, it is the standard procedure in the industry.

For each client, the difference in base values for both models is taken as the "base NPS value," and the difference in Shapley values per variable is associated with an "NPS Shap value." The logic is simple, and since the operations are linear, it can be considered correct.

To compute the explainability of the NPS for a sample of clients, the NPS base value is taken and each NPS Shap value is averaged separately.

## Explanation of difference as a difference in explanations.

The end goal of the tool is to compare and explain why there is a difference in NPS between two periods of time (or two samples of clients). The main assumption is that this explanation is somehow hidden in the difference (not in a mathematical sense) in NPS explanations for each period of time. For now, the mathematical difference is taken as a proxy to extract this information.

After computing the NPS and its explainability with the above procedure for two periods of time, the result of subtracting the Shapley values separately is taken as the explanation of why the NPS changed. Notice that because the models remain the same, when the base values of both samples are subtracted, they cancel out.

One thing that is quickly noticeable using this approach is how, due to both the non-linear nature of the model and the uncertainty in the explanations, there are sometimes variables that have a positive change and are known to have a positive impact but are counterintuitively translated into negative Shapley values.

## Uncertainty: MAE and flipped shaps.

![VS Past Sources of Uncertainty](src/VS_sources_uncertainty.png)
