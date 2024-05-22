# Placeholder Name GitHub Repository
- Team Members: Bella, Blaize, Monty, Nikita, Will
 - We placed 37th globally (1st in Australia) in the 2024 IMC Prosperity Competition.

### Round 1
- Very similar to both the tutorial and the first round of last years competition. We used market making / taking strategies on both amtheysts and starruit. Amethysts were simple as the 'fair price' was given at 10,000 and there was very little fluctuation from here, whereas for starfruit we implemented a simple linear regression in order to predict the fair price based off of previous price points.
- After this round we were a few hundred places from the top but the leaderboard was so tight we didn't see this as cause for concern. Manual trading was trivial.

### Round 2
- Made small changes to Round 1 algorithm improving it by ~10%
- The product introduced in this round was orchids, produced on the southern acrhipelago and thus can be imported from and exported to there. We found an arbitrage strategy in which we sold orchids locally before purchasing them from the south archipelago on the next tick in order to make a guaranteed profit. This was very effective due to the negative import tariff (meaning we were given seashells for importing orchids).
- After this round we moved up into 44th position, giving us a fighting chance at some prizes.

### Round 3
- The product introduced in this round was gift baskets, as long as their components. We used statistical arbitrage, using the size of the gift basket premium as a signal for when to buy and sell. We also chose to only trade gift baskets as they gave consistently positive returns (we believe this to be because of the component prices being a leading indicator for gift basket prices). We also implemented a strategy for trading roses similar to pair trading that gave good results in the backtester, however we ended up losing ~10k on submission so we removed this for future rounds.
- We moved up 3 places after this round into 41st which was benefitted by a very competitive manual trading score.

### Round 4
- This round saw the introduction of call options for coconuts. The strategy we devised used the Black-Scholes option pricing model to calculate a fair price for the options, and then trading whenever this deviated significantly from the market price, believing the theoretical and market price should eventually converge again.
- This round was very successful and we additionally did quite well on manual trading moving us from 41st to 28th position, just 3 places out of the prizes...

### Round 5
- This round saw no new products, just more data given revealing the counterparty for each trade. We scoured the data and only found one reliable signal (Rhianna consistently bought roses low and sold them high), so we simply devised a strategy to copy her trades. As for the other bots we were unable to find any consistent strategies using this new data and thus we decided to marginally improve our algorithm in other areas, including orchids which had performed badly in the previous round. Following the final round we placed 37th overall.