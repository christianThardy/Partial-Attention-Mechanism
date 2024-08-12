# Partial Attention Large Language Model

This implementation of **PALLM** takes the strong foundation of decoder-only models and enhances them with the ability to focus on what's important, remember context, and maintain coherent, on-topic conversations.

It's essentially taking the best thing about encoder-decoder models (bidirectionality to capture rich contextual representations) and baking them directly into a decoder-only model via a partial attention mechanism 
that alleviates attention degeneration, a bidirectional attention mask, a separate positional encoding and a specialized language embedding to help the model differentiate between source (prompt) and target 
(generated output) text parts/sequences. PALLM allows:

  - **More Coherent Responses for Nuanced Tasks:**
    - By always keeping the main topic in focus, PALLM generates responses that stay more relevant to the original question or prompt.
      
  - **Reduced "Forgetting":**
    - Long conversations are less likely to go off-track because the model maintains a strong connection to the initial prompt.
      
  - **Better Context Understanding:**
    - The model can better differentiate between the user's input and its own responses, leading to better back-and-forth exchanges that stay on topic.
      
  - **Improved Long-Form Generation:**
    - For tasks that require longer outputs, like storytelling, detailed explanations or dialogue, PALLM helps maintain consistency and relevance throughout.

<br>

### Reference:

Fu, Lam, Yu, Cho So, Hu, Liu,  Collier, *Decoder-Only or Encoder-Decoder? Interpreting Language Model as a Regularized Encoder-Decoder*. 2023. [<a href="https://arxiv.org/pdf/2304.04052" rel="nofollow">1</a></li>]
