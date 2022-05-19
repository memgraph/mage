CREATE (:Stock {name: "AMZN", id: 0});
CREATE (:Stock {name: "AAPL", id: 1});
CREATE (:Stock {name: "ZION", id: 2});
CREATE (:Stock {name: "ABT", id: 3});
CREATE (:Stock {name: "TSLA", id: 4});
CREATE (:Stock {name: "MSFT", id: 5});

CREATE (:TradingDay {date: "2022-04-23", id: 0});
CREATE (:TradingDay {date: "2022-04-24", id: 1});
CREATE (:TradingDay {date: "2022-04-25", id: 2});
CREATE (:TradingDay {date: "2022-04-26", id: 3});

MATCH (a:Stock {id: 0}), (b:TradingDay {id: 0}) CREATE (a)-[e:Traded_On {open: 120, close: 145, high:146, low: 117, volume: 50000}]->(b);
MATCH (a:Stock {id: 1}), (b:TradingDay {id: 0}) CREATE (a)-[e:Traded_On {open: 100, close: 92, high:100, low: 91, volume: 50000}]->(b);
MATCH (a:Stock {id: 2}), (b:TradingDay {id: 0}) CREATE (a)-[e:Traded_On {open: 120, close: 145, high:146, low: 117, volume: 50000}]->(b);
MATCH (a:Stock {id: 3}), (b:TradingDay {id: 0}) CREATE (a)-[e:Traded_On {open: 100, close: 92, high:100, low: 91, volume: 50000}]->(b);
MATCH (a:Stock {id: 4}), (b:TradingDay {id: 0}) CREATE (a)-[e:Traded_On {open: 120, close: 145, high:146, low: 117, volume: 50000}]->(b);
MATCH (a:Stock {id: 5}), (b:TradingDay {id: 0}) CREATE (a)-[e:Traded_On {open: 100, close: 120, high:250, low: 100, volume: 50000}]->(b);

MATCH (a:Stock {id: 0}), (b:TradingDay {id: 1}) CREATE (a)-[e:Traded_On {open: 120, close: 145, high:146, low: 117, volume: 50000}]->(b);
MATCH (a:Stock {id: 1}), (b:TradingDay {id: 1}) CREATE (a)-[e:Traded_On {open: 100, close: 92, high:100, low: 91, volume: 50000}]->(b);
MATCH (a:Stock {id: 2}), (b:TradingDay {id: 1}) CREATE (a)-[e:Traded_On {open: 120, close: 145, high:146, low: 117, volume: 50000}]->(b);
MATCH (a:Stock {id: 3}), (b:TradingDay {id: 1}) CREATE (a)-[e:Traded_On {open: 100, close: 92, high:100, low: 91, volume: 50000}]->(b);
MATCH (a:Stock {id: 4}), (b:TradingDay {id: 1}) CREATE (a)-[e:Traded_On {open: 120, close: 145, high:146, low: 117, volume: 50000}]->(b);
MATCH (a:Stock {id: 5}), (b:TradingDay {id: 1}) CREATE (a)-[e:Traded_On {open: 100, close: 120, high:250, low: 100, volume: 50000}]->(b);

MATCH (a:Stock {id: 0}), (b:TradingDay {id: 2}) CREATE (a)-[e:Traded_On {open: 120, close: 145, high:146, low: 117, volume: 50000}]->(b);
MATCH (a:Stock {id: 1}), (b:TradingDay {id: 2}) CREATE (a)-[e:Traded_On {open: 100, close: 92, high:100, low: 91, volume: 50000}]->(b);
MATCH (a:Stock {id: 2}), (b:TradingDay {id: 2}) CREATE (a)-[e:Traded_On {open: 120, close: 145, high:146, low: 117, volume: 50000}]->(b);
MATCH (a:Stock {id: 3}), (b:TradingDay {id: 2}) CREATE (a)-[e:Traded_On {open: 100, close: 92, high:100, low: 91, volume: 50000}]->(b);
MATCH (a:Stock {id: 4}), (b:TradingDay {id: 2}) CREATE (a)-[e:Traded_On {open: 120, close: 145, high:146, low: 117, volume: 50000}]->(b);
MATCH (a:Stock {id: 5}), (b:TradingDay {id: 2}) CREATE (a)-[e:Traded_On {open: 100, close: 120, high:250, low: 100, volume: 50000}]->(b);

MATCH (a:Stock {id: 0}), (b:TradingDay {id: 3}) CREATE (a)-[e:Traded_On {open: 120, close: 145, high:146, low: 117, volume: 50000}]->(b);
MATCH (a:Stock {id: 1}), (b:TradingDay {id: 3}) CREATE (a)-[e:Traded_On {open: 100, close: 92, high:100, low: 91, volume: 50000}]->(b);
MATCH (a:Stock {id: 2}), (b:TradingDay {id: 3}) CREATE (a)-[e:Traded_On {open: 120, close: 145, high:146, low: 117, volume: 50000}]->(b);
MATCH (a:Stock {id: 3}), (b:TradingDay {id: 3}) CREATE (a)-[e:Traded_On {open: 100, close: 92, high:100, low: 91, volume: 50000}]->(b);
MATCH (a:Stock {id: 4}), (b:TradingDay {id: 3}) CREATE (a)-[e:Traded_On {open: 120, close: 145, high:146, low: 117, volume: 50000}]->(b);
MATCH (a:Stock {id: 5}), (b:TradingDay {id: 3}) CREATE (a)-[e:Traded_On {open: 100, close: 120, high:250, low: 100, volume: 50000}]->(b);
