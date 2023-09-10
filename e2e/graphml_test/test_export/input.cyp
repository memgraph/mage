queries: CREATE (StrangerThings:TVShow {title:'Stranger Things', released:2016, program_creators:['Matt Duffer', 'Ross Duffer']}), (Eleven:Character {name:'Eleven', portrayed_by:'Millie Bobby Brown'}), (JoyceByers:Character {name:'Joyce Byers', portrayed_by:'Millie Bobby Brown'}), (JimHopper:Character {name:'Jim Hopper', portrayed_by:'Millie Bobby Brown'}), (MikeWheeler:Character {name:'Mike Wheeler', portrayed_by:'Finn Wolfhard'}), (DustinHenderson:Character {name:'Dustin Henderson', portrayed_by:'Gaten Matarazzo'}), (LucasSinclair:Character {name:'Lucas Sinclair', portrayed_by:'Caleb McLaughlin'}), (NancyWheeler:Character {name:'Nancy Wheeler', portrayed_by:'Natalia Dyer'}), (JonathanByers:Character {name:'Jonathan Byers', portrayed_by:'Charlie Heaton'}), (WillByers:Character {name:'Will Byers', portrayed_by:'Noah Schnapp'}), (SteveHarrington:Character {name:'Steve Harrington', portrayed_by:'Joe Keery'}), (MaxMayfield:Character {name:'Max Mayfield', portrayed_by:'Sadie Sink'}), (RobinBuckley:Character {name:'Robin Buckley', portrayed_by:'Maya Hawke'}), (EricaSinclair:Character {name:'Erica Sinclair', portrayed_by:'Priah Ferguson'}), (Eleven)-[:ACTED_IN {seasons:[1, 2, 3, 4]}]->(StrangerThings), (JoyceByers)-[:ACTED_IN {seasons:[1, 2, 3, 4]}]->(StrangerThings), (JimHopper)-[:ACTED_IN {seasons:[1, 2, 3, 4]}]->(StrangerThings), (MikeWheeler)-[:ACTED_IN {seasons:[1, 2, 3, 4]}]->(StrangerThings), (DustinHenderson)-[:ACTED_IN {seasons:[1, 2, 3, 4]}]->(StrangerThings), (LucasSinclair)-[:ACTED_IN {seasons:[1, 2, 3, 4]}]->(StrangerThings), (NancyWheeler)-[:ACTED_IN {seasons:[1, 2, 3, 4]}]->(StrangerThings), (JonathanByers)-[:ACTED_IN {seasons:[1, 2, 3, 4]}]->(StrangerThings), (WillByers)-[:ACTED_IN {seasons:[1, 2, 3, 4]}]->(StrangerThings), (SteveHarrington)-[:ACTED_IN {seasons:[1, 2, 3, 4]}]->(StrangerThings), (MaxMayfield)-[:ACTED_IN {seasons:[2, 3, 4]}]->(StrangerThings), (RobinBuckley)-[:ACTED_IN {seasons:[3, 4]}]->(StrangerThings), (EricaSinclair)-[:ACTED_IN {seasons:[2, 3, 4]}]->(StrangerThings);
nodes: MATCH (a) RETURN a;
relationships: MATCH (a) -[r]-> (b) RETURN r;
