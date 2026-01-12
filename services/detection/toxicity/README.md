Automated vulnerability discovery service based on technique introduced in:

Ethan Perez, Saffron Huang, Francis Song, Trevor Cai, Roman Ring, John
Aslanides, Amelia Glaese, Nat McAleese, and Geoffrey Irving. 2022. Red Teaming
Language Models with Language Models. In Proceedings of the 2022 Conference on
Empirical Methods in Natural Language Processing, pages 3419–3448, Abu Dhabi,
United Arab Emirates. Association for Computational Linguistics.

Generic structure of method:

1. Generate many test cases using a red LM.
2. Use the target LM to generate an output y for each test case x.
3. Find the test cases that led to a harmful output using the red team
   classifier r(x, y).
