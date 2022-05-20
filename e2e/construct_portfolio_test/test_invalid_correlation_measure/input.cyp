CREATE (s:Stock {name: "AMZN", id: 0});
CREATE (s:Stock {name: "AAPL", id: 1});
CREATE (s:Stock {name: "ZION", id: 2});
CREATE (s:Stock {name: "ABT", id: 3});
CREATE (s:Stock {name: "TSLA", id: 4});
CREATE (s:Stock {name: "MSFT", id: 5});
CREATE (d:TradingDay {date: "2022-04-23", id: 10});
CREATE (d:TradingDay {date: "2022-04-24", id: 11});
CREATE (d:TradingDay {date: "2022-04-25", id: 12});
CREATE (d:TradingDay {date: "2022-04-26", id: 13});
MATCH (a:Stock {id: 0}), (b:TradingDay {id: 10}) CREATE (a)-[e:Traded_On {open: 120.0, close: 145.0, high:146.0, low: 117.0, volume: 50000}]->(b);
MATCH (a:Stock {id: 1}), (b:TradingDay {id: 10}) CREATE (a)-[e:Traded_On {open: 100.0, close: 92.0, high:100.0, low: 91.0, volume: 50000}]->(b);
MATCH (a:Stock {id: 2}), (b:TradingDay {id: 10}) CREATE (a)-[e:Traded_On {open: 120.0, close: 145.0, high:146.0, low: 117.0, volume: 50000}]->(b);
MATCH (a:Stock {id: 3}), (b:TradingDay {id: 10}) CREATE (a)-[e:Traded_On {open: 100.0, close: 92.0, high:100.0, low: 91.0, volume: 50000}]->(b);
MATCH (a:Stock {id: 4}), (b:TradingDay {id: 10}) CREATE (a)-[e:Traded_On {open: 120.0, close: 145.0, high:146.0, low: 117.0, volume: 50000}]->(b);
MATCH (a:Stock {id: 5}), (b:TradingDay {id: 10}) CREATE (a)-[e:Traded_On {open: 100.0, close: 120.0, high:250.0, low: 100.0, volume: 50000}]->(b);
MATCH (a:Stock {id: 0}), (b:TradingDay {id: 11}) CREATE (a)-[e:Traded_On {open: 120.0, close: 145.0, high:146.0, low: 117.0, volume: 50000}]->(b);
MATCH (a:Stock {id: 1}), (b:TradingDay {id: 11}) CREATE (a)-[e:Traded_On {open: 100.0, close: 92.0, high:100.0, low: 91.0, volume: 50000}]->(b);
MATCH (a:Stock {id: 2}), (b:TradingDay {id: 11}) CREATE (a)-[e:Traded_On {open: 120.0, close: 145.0, high:146.0, low: 117.0, volume: 50000}]->(b);
MATCH (a:Stock {id: 3}), (b:TradingDay {id: 11}) CREATE (a)-[e:Traded_On {open: 100.0, close: 92.0, high:100.0, low: 91.0, volume: 50000}]->(b);
MATCH (a:Stock {id: 4}), (b:TradingDay {id: 11}) CREATE (a)-[e:Traded_On {open: 120.0, close: 145.0, high:146.0, low: 117.0, volume: 50000}]->(b);
MATCH (a:Stock {id: 5}), (b:TradingDay {id: 11}) CREATE (a)-[e:Traded_On {open: 100.0, close: 120.0, high:250.0, low: 100.0, volume: 50000}]->(b);
MATCH (a:Stock {id: 0}), (b:TradingDay {id: 12}) CREATE (a)-[e:Traded_On {open: 120.0, close: 145.0, high:146.0, low: 117.0, volume: 50000}]->(b);
MATCH (a:Stock {id: 1}), (b:TradingDay {id: 12}) CREATE (a)-[e:Traded_On {open: 100.0, close: 92.0, high:100.0, low: 91.0, volume: 50000}]->(b);
MATCH (a:Stock {id: 2}), (b:TradingDay {id: 12}) CREATE (a)-[e:Traded_On {open: 120.0, close: 145.0, high:146.0, low: 117.0, volume: 50000}]->(b);
MATCH (a:Stock {id: 3}), (b:TradingDay {id: 12}) CREATE (a)-[e:Traded_On {open: 100.0, close: 92.0, high:100.0, low: 91.0, volume: 50000}]->(b);
MATCH (a:Stock {id: 4}), (b:TradingDay {id: 12}) CREATE (a)-[e:Traded_On {open: 120.0, close: 145.0, high:146.0, low: 117.0, volume: 50000}]->(b);
MATCH (a:Stock {id: 5}), (b:TradingDay {id: 12}) CREATE (a)-[e:Traded_On {open: 100.0, close: 120.0, high:250.0, low: 100.0, volume: 50000}]->(b);
MATCH (a:Stock {id: 0}), (b:TradingDay {id: 13}) CREATE (a)-[e:Traded_On {open: 120.0, close: 145.0, high:146.0, low: 117.0, volume: 50000}]->(b);
MATCH (a:Stock {id: 1}), (b:TradingDay {id: 13}) CREATE (a)-[e:Traded_On {open: 100.0, close: 92.0, high:100.0, low: 91.0, volume: 50000}]->(b);
MATCH (a:Stock {id: 2}), (b:TradingDay {id: 13}) CREATE (a)-[e:Traded_On {open: 120.0, close: 145.0, high:146.0, low: 117.0, volume: 50000}]->(b);
MATCH (a:Stock {id: 3}), (b:TradingDay {id: 13}) CREATE (a)-[e:Traded_On {open: 100.0, close: 92.0, high:100.0, low: 91.0, volume: 50000}]->(b);
MATCH (a:Stock {id: 4}), (b:TradingDay {id: 13}) CREATE (a)-[e:Traded_On {open: 120.0, close: 145.0, high:146.0, low: 117.0, volume: 50000}]->(b);
MATCH (a:Stock {id: 5}), (b:TradingDay {id: 13}) CREATE (a)-[e:Traded_On {open: 100.0, close: 120.0, high:250.0, low: 100.0, volume: 50000}]->(b);
