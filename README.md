# Partial Attention Large Language Model

This ongoing implementation of **PALM** is a wrapper on top of HuggingFace models to take the strong foundation of decoder-only models and enhance them to use bidirectionality to capture rich contextual representations (like an encoder-decoder model) via a partial attention mechanism that decreases the effects of attention degeneration as seen in the hallucination problem via a bidirectional attention mask, a separate positional encoding and a specialized language embedding to help the model differentiate between source (prompt) and target (generated output) text parts/sequences. PALM allows:

  - **More Coherent Responses for Nuanced, Long-form Tasks:**
    - By always keeping the main topic in focus, PALM generates responses that stay more relevant to the original prompt.
      
  - **Reduced "Forgetting":**
    - Long conversations are less likely to go off-track because the model maintains a strong connection to the initial prompt.
      
  - **Better Context Understanding:**
    - The model can better differentiate between the user's input and its own responses, leading to better back-and-forth exchanges that stay on topic.
      
  - **Improved Long-Form Generation:**
    - For tasks that require longer outputs, like storytelling, detailed factual explanations or dialogue, PALM helps maintain consistency and relevance throughout.

<br>

### Reference:

Fu, Lam, Yu, Cho So, Hu, Liu,  Collier, *Decoder-Only or Encoder-Decoder? Interpreting Language Model as a Regularized Encoder-Decoder*. 2023. [<a href="https://arxiv.org/pdf/2304.04052" rel="nofollow">1</a></li>]
