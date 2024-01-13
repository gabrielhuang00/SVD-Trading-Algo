# SVD-Trading-Algo
Systems Trading Algorithm developed in Python's Backtrader API.

This project was developed based on the inspiration to use negative correlation between market sectors as a trading signal.
The technique used involved SVD, or Singular Value Decomposition. SVD allowed me to break down market data (stored in the form of a matrix) into basis vectors.
Of course, the first basis vector revealed positive correlation between all sectors (Since the primary basis vector reveals the strongest correlation, and all
stocks have, on average, trended up in the last 30 years). 

Further analysis found interesting contrasts between different sectors. When put into a backtesting API on Python, results showed little to no alpha. Still, 
a super cool project and fun to put together.
