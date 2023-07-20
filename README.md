# NBA-Dataset-Analysis

## Introduction

For this analysis, I wanted to use an NBA dataset for my project. The dataset contains very simple data regarding NBA games and NBA players who played that game. There are no advanced stats so I wont be able to dive extremely deep into the analysis, but the simple stats are enough to draw some conclusions and some correlations. I intend to draw statistical conclusions such as how certain stats are related to one another, how the game differs in the playoffs compared to the regular season, and how the game changed in the last decade. I also intend to create some machine learning algorithms that can pick up on some patterns and output some correlations that it finds.

## The NBA Dataset Ethical Audit

The NBA Dataset contains multiple csv files each having their own theme. One of the files for example, is called games.csv and it contains team data (such as how many points a team scored, who won, how many points the opponent scored and these games can be traced back all the way to the 2013 season.There are a few consideration that should be considered when using this dataset however.
The first is because there are multiple files you are going to have to connect the two files to get the full picture. So for example one file would contain a player name and a player id and another file will contain all players stats that a player had in that game.
Another thing is this dataset does not contain every single piece of data of the nba. It does not have a player's wingspan, or height. It does not contain any advanced stats. It more focuses on the simple statistics such as how many points a player/team scored, how many assists did a player have, how well did he shoot the ball, ect.
According to the last piece of record on this dataset, I can make the assumption that the person who created this dataset created it in 2022-12-22 because that is the last piece of data which can be seen in this dataset. This makes it a bit unfortunate because that means I can not use the data from the 2022-2023 season, due to the fact that it was not complete.
An additional analysis revealed this dataset does not split the regular season games, playoffs or preseason games. Instead it simply stores it all under the same files so in order to differentiate between the 3, I have to manually put in the dates for which time the regular season started because I am going to be doing my analysis primarily on the regular season due to its large data sample.

## Exploratory Analysis

When first analyzing the data, I first wanted to see if there were any null values. To my surprise there was a great deal of missing null values in the games_detail file which is a file that contains all the information in that game such as which player played, what was his position, his stats, the team name, ect. There appeared to be 558938 out of 668628 non null values from the statistics part of the dataframe which tells me that the players who did not play a minute of basketball, but were marked as present in the game simply had their row filled up as null. 
Then there were columns which had a major deal of missing values. These columns were NICKNAME, START_POSITION,  and COMMENT. The NICKNAME column is self explanatory, not every player is going to have a nickname so it's left as null. The START_POSITION can be explained as players who did not start in the game instead were bench players. This would result in their values being null instead of the 3 categorical values C (Center), F (Forward), G (Guard). This would become an issue later on when I would want to do one of my analyses which required the position of the player. Lastly, the COMMENT column represents a reason why a player did not play that game. So if a player did not play, the coach would leave a ‘comment’. With these columns it's not a big deal I can simply drop the column due to it not being useful for any statistical analysis, besides the STAT_POSITION which could be useful but not for correlational data.
The last piece of data which I found interesting was for the PLUS_MINUS which is a statistical measurement of a player’s performance, there seems to be 535,277  non null values out of the possible 558,938. This means around 23,000 data is missing from the player statistic which is interesting as to why that is so.

![NullValues](https://github.com/eitancohen77/NBA-Dataset-Analysis/assets/98838116/93164f90-50db-4a3c-b668-2b9ad533f4f7)

As for the numeric values in this dataset, I am only going to focus on statistical measurements such as team stats and individual player stats. For this I am going to explain what each variable represents

<h3 align="center">Indvidual Player Stats</h3>

- MIN: How many minutes a player played.
- FGM: How many shots a player made (includes 3 point makes)
- FGA: How many shots a player attempted (includes 3 point attempts)
- FG3M: How many 3 point shots a player made.
- FG3A: How many 3 point shots a player attempted.
- FTM: How many free throws a player made
- FTA: How many free throws a player attempted
- FT_PCT: FTM/FTA
- OREB: Offensive Rebounds. This occurs when the offensive team (the one trying to score a basket) misses a shot, but get the rebound and therefor get a second chance to score
- DREB: Defensive Rebound. This occurs when the offensive team misses a shot and the defensive team gets the rebound.
- REB: OREB + DREB
- AST: Assists. This happens when a player "assisted" another player and passed him the ball to score.
- STL: Steals. This happens when a defensive player steals the ball from an offensive player.
- BLK: Blocks. This happens when a defensive player blocks a shot attempt from an offensive player
- GP: Games Played. This showcases how many games a player played that season,

<h3 align="center">Team Stats</h3>

- PTS_home: how many points the home team scored that game
- FG_PCT_home: what was the shooting percentage the home team shot that night. So if they shot 10 times and made 6, there FG_PCT would be 60% or 0.60
- FT_PCT_home: the free throw percentage the home team shot in that game. If they shot 10 free throws and made 8, there FT_PCT would be 0.80
- FG3_PCT_home: the percentage of a 3 point attempt
- (All the aways columns are the same as the home except its for the other team that game)
- PLUS_MINUS: A stat that measures how much impact a player had since he got on the court. Did the score change in a positive or a negative direction?


Doing a little quartile analysis, I found out that the league averages in minutes per game is around 19 and the average points per game is 8.2. The minutes do however increase to 27 minutes a game when it comes to the top 75% players of the league.
Another interesting statistic I saw was when I grouped by all the players' positions to see how many starting players in each position were there in the league. There seemed to be a correlation between these positions as centers had a 1:5 ratio, forwards had a 2:5 ratio and guards also had a 2:5 ratio. This does however make sense because in the NBA there are 5 positions. 2 of which are guards, 2 are forwards and 1 is center so it's cool to see that statistic visualized.

![](https://github.com/eitancohen77/NBA-Dataset-Analysis/assets/98838116/fca01b72-6d0b-4800-812f-f835ab69a90e)
![](https://github.com/eitancohen77/NBA-Dataset-Analysis/assets/98838116/b94b8bf3-add3-4fd3-8737-59d50a124395)
![](https://github.com/eitancohen77/NBA-Dataset-Analysis/assets/98838116/79876c10-c2c5-4a09-b317-6bec12c0794f)


# High Level Analysis

## Analysis 1: Statistical Correlation between Player Stats
With this analysis, I want to see if there is any correlation between individual player stats. I will be using the stats from players of the 2021-2022 regular season. In order to make this correlation I want to make some constraints in order to make this fair. The first one is to divide the players total stats for that season by the total minutes they played instead of the games they played. This makes it fair because I will be able to see how productive they were in the minutes they played. 

![](https://github.com/eitancohen77/NBA-Dataset-Analysis/assets/98838116/a7efb90d-856d-4cf3-b54c-1a1358cb7416)

Another constraint I want to add is players who played very few minutes. The reason why I want to do this is because in the NBA, we have this thing called garbage time where essentially if a team is up by a substantial amount of points, it will sub in non valuable players as to not risk the injury of their better star players. Another reason why I want this is because there could players in the league who played only 5 minutes that entire season and had no stats. This would weigh down the whole correlation matrix.

![](https://github.com/eitancohen77/NBA-Dataset-Analysis/assets/98838116/7fbb3bed-5b76-442e-bb92-20789e88ef13)
- As we can see only 81% of the league played more than 100 minutes in the league so I would want this statistic and I get the heatmap:


![heatmap](https://github.com/eitancohen77/NBA-Dataset-Analysis/assets/98838116/d140af81-6283-4328-9c7e-c8f08d23ac8f)
![heatmapContinuation](https://github.com/eitancohen77/NBA-Dataset-Analysis/assets/98838116/58bde33f-8df2-4b04-8622-a72698bfc3d7)

Some interesting stats I found from this heatmap:
Players who rebound well, tend to not shoot the ball from the 3 point line. This does make sense because players who rebound a lot are mostly centers and these are the players who will be around the rim, far away from the 3 point line.
Players who attempt 3 pointers have their FG_PCT decrease which is their overall field goal attempts. This makes sense because a 3 point attempt is more difficult to make.
Players who have a lot of PF (personal fouls) tend to not take a lot of 3s, and tend to have a lot of REB (rebounds). This leans into the idea that centers tend to get more physical when fighting for the rebound. 
An interesting stat that I found was there seems to be a somewhat positive correlation between assists and steals. My guess for this is guards who are the playmakers responsible for assists would need to use their hands more resulting in steals.
<br><br><br>
To test these how significantly these correlations are we can use PearsonR test:
![](https://github.com/eitancohen77/NBA-Dataset-Analysis/assets/98838116/26b432fc-0063-442b-90de-454a4e9482ee)
<br><br><br>

## Analysis 2: Distribution of Minutes in the NBA
