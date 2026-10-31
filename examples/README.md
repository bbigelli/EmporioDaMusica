# Interaction examples

Generated with `python -m emporio.cli demo`, so they can be reproduced rather than taken on trust.

Tool calls and their raw results are folded into each transcript. In a customer-facing product they would be hidden; here they are the point — they show whether the agent consulted the catalog, the policy manual, both, or neither.

- [Browsing the catalog on a budget](01-catalog-on-a-budget.md)  
  Structured catalog query with a price ceiling, then a follow-up that depends on remembering which instrument was being discussed. Also shows how unavailable matches are handled: two of the fourteen acoustic guitars under BRL 1000 are out of stock or discontinued, and the agent should not present them as buyable.

- [A promotional price meets the PIX discount](02-promotional-price-and-pix.md)  
  The non-trivial pricing case. The Crafter HT-100 carries an active 18% discount, and section 6.2 says promotions are not cumulative — so the 5% PIX discount does *not* stack on top of it. Answering this correctly requires reading the live promotion and applying the rule; a plausible wrong answer (R$ 1.868,82) undercharges the store.

- [The withdrawal window closed, but the warranty has not](03-return-window-expired.md)  
  The non-trivial policy case, and the reason the rules layer reports all four return routes instead of one verdict. Order 7 was delivered 2026-02-17, so against the reference date of 2026-03-25 the 7-day withdrawal, the 30-day defect exchange and the 7-day preference exchange have all expired — but the 90-day legal warranty runs to 2026-05-18. A flat 'no' would be both unhelpful and wrong.

- [Two kinds of out of scope](04-out-of-scope.md)  
  Section 7.1 draws a line the agent has to hold: the store sells instruments and explicitly not accessories, so a request for strings gets a warm redirect rather than a catalog search. The second question is outside the store entirely and should get one light sentence, not a lecture about limitations. The third checks the agent comes back to work afterwards.

- [An English-speaking customer, shipping outside the region](05-english-customer-shipping.md)  
  Checks the two things that interact badly if either is wrong. The agent must mirror the customer's language, and it must not invent a shipping fee: São Paulo is outside metropolitan Campo Grande, where the cost depends on postcode, weight and dimensions. Carrier names and estimates are available; a number is not. The Taylor 110e also carries an active 8% discount, which should be presented with its original price.
