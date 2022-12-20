# DOUBTS GLOSSARY

In this document, the team doubts and solutions will be documented.

## MDP rule related

- **DOUBT:** If we use the distance run from the last state as an input, is it going against the MDP principle?
   " *Just the current state must be considered to perform an action since the previous state does not affect to the current
  decision* " 
  - **RESPONSE** Once you are in a given state, it doesn't matter how you got there. The current state contains all the information required to make a prediction or a decision. This simplifies the decision algorithm a lot, because it does not need any memory.
  That said, your input could be any kind of information that gives you an idea of the environment state, so, as long as this input does no imply a states explosion, it should be fine.