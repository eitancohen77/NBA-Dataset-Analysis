# NBA-Dataset-Analysis

## Introduction

For this analysis, I wanted to use an NBA dataset for my project. The dataset contains very simple data regarding NBA games and NBA players who played that game. There are no advanced stats so I wont be able to dive extremely deep into the analysis, but the simple stats are enough to draw some conclusions and some correlations. I intend to draw statistical conclusions such as how certain stats are related to one another, how the game differs in the playoffs compared to the regular season, and how the game changed in the last decade. I also intend to create some machine learning algorithms that can pick up on some patterns and output some correlations that it finds.

## The NBA Dataset Ethical Audit

The NBA Dataset contains multiple csv files each having their own theme. One of the files for example, is called games.csv and it contains team data (such as how many points a team scored, who won, how many points the opponent scored and these games can be traced back all the way to the 2013 season. There are a few things that should be considered when using this dataset however.
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
Out of 608 players, around 180 play between 0 - 250 minutes. That would mean around 30% of players in the NBA don't play that many minutes. This also does not even take into account players who did not even step foot in the nba.
![](https://github.com/eitancohen77/NBA-Dataset-Analysis/assets/98838116/f3a29a5c-d2dd-41dc-ac2f-bcadfab9737a)

<br>
I am curious to see how this will dwell against the playoffs. I would imagine the extremes get worse because the good players would play more minutes, while the bad players probably don't step foot on the court. This leads me to my hypothesis:I believe that in the playoffs star players play a lot more minutes then if they were to play in the regular season, while bench players play less minutes in the playoffs than the regular season.
To view this visualization, I am going to calculate based on minutes played per game.

![](https://github.com/eitancohen77/NBA-Dataset-Analysis/assets/98838116/a1c2c67c-6bc2-441a-9a41-2573380a2f15)
![](https://github.com/eitancohen77/NBA-Dataset-Analysis/assets/98838116/8aa77393-bce3-40e6-9980-dc8a91c6b48d)

- As I expected, when the playoffs enter, the minutes for distribution starts skewing to the 2 extremes. From this we can say a cause for this occurrence is the coach needs  his best players to play while the role players play less. 

We can also test the averages of these 2 by using a T-test
![](https://github.com/eitancohen77/NBA-Dataset-Analysis/assets/98838116/b9aaa004-6ccf-429d-99ab-f6c0f42af859)

![](https://github.com/eitancohen77/NBA-Dataset-Analysis/assets/98838116/c0148735-0f72-466b-8d87-b499091639cf)

Average minutes per game across the NBA in the regular season is 18.9 whereas in the playoffs it's 19.4. The difference is statistically insignificant at p = 0.53.
<br><br><br>

## Analysis 3: Game Change in the Past 9 Years
In the basketball world, we have a saying that Stephen Curry revolutionized the game of basketball. He showed the world that if you are really good at shooting, you can attempt to shoot the basketball from 30 feet beyond the rim. This ‘revolution’ started around his first MVP season in the 2014-2015 season. I want to track  each season from the 2013-2014 all the way until the 2021-2022 season to see how much the game changed thanks to Stephen.
![](https://github.com/eitancohen77/NBA-Dataset-Analysis/assets/98838116/10c78531-17fb-4e15-8122-d5a9b18eb4e6)

Then I want to convert this into averages
![](https://github.com/eitancohen77/NBA-Dataset-Analysis/assets/98838116/c739b490-27a2-4db5-9dea-2a73e71abed5)

Some of the things I want to see if there was a progression is the 3 point shot, total number of possessions, and true shooting percentage. 

<h3 align="center">3 Point Attempt</h3>

![](https://github.com/eitancohen77/NBA-Dataset-Analysis/assets/98838116/9663d9ba-4367-4784-868f-db8d814740a8)

- We can see a very noticeable trend in this case. We can use a Linear regression test to see how significant this data is.
![](https://github.com/eitancohen77/NBA-Dataset-Analysis/assets/98838116/3f935b65-4957-4097-80ce-229c2e2012b2)

- From this linear regression stat, we can see that the p value is significantly less than 0.01 meaning there is significant evidence for strong correlation between 3 point attempts as time progresses, with an average of about 2 more 3s being attempted every year since 2013.

<h3 align="center">Possessions</h3>
Some interesting stats I noticed is how points increased, assists increased, FGA increased, and turnover decreased. This led me to think that possessions might be increasing as well. Possessions is a stat that is incremented every time an opponent touches the ball. This results in a faster pace of playing because the time in a game did not increase as its been 48 minutes for 76 years. To calculate possessions you do the formula (FGA - OR) + TO + (Y * FTA), where FGA is field goal attempts, OR is offensive rebounds, TO is turnovers, Y is some number between zero and 1 (most people use 0.44) and FTA is free throw attempts. Reason why people use 0.44 is because that's how much analysts believe a free throw is statistically worth.

![](https://github.com/eitancohen77/NBA-Dataset-Analysis/assets/98838116/9d6c212e-3f99-48ad-9c6e-b08f54aca755)

<br>
We can use a linear regression to test if this is significant.

![](https://github.com/eitancohen77/NBA-Dataset-Analysis/assets/98838116/579722e0-39ae-4bd0-9452-70b15af28aee)

- From the linear regression test, we can say that the p value is less than 0.01 meaning there is significant evidence for correlation between possessions as the NBA progresses since 2013 with an average of 0.8 possessions increasing each year. With this we can say that the NBA pacing is currently getting faster.

<h3 align="center">True Shooting Percentage</h3>
I want to measure how better the players are getting at shooting. To test this, I am going to need an advanced stat which I can calculate. This stat is called the True Shooting Percentage and it measures how efficient a player is when it comes to shooting the basketball. If the true shooting percentage goes up then I can say that players are becoming better at shooting the basketball. To calculate true shooting percentage you do (PTS / ( 2*FGA + 0.44*FTA)).

![](https://github.com/eitancohen77/NBA-Dataset-Analysis/assets/98838116/aea3580e-698a-4f2b-b801-fcf136c32803)

<br>
We can use the linear regression test here as well.

![](https://github.com/eitancohen77/NBA-Dataset-Analysis/assets/98838116/493b2cd5-e635-4a45-94e5-daf8559f534b)

From this linear regression test, we can see that the p value  is less than 0.01 meaning there is evidence for correlation for players having their true shooting percentage numbers increase as they progress with an average of 0.004 increase each year since 2013.

You can argue that Curry was not the sole purpose for this change in game and that it was headed that way regardless and all Curry did was simply spark the fire. I am more interested to see the stats from a while back because from what these charts show, it started off being flat and then when Curry came in 2014, the pacing increased. I wonder if there was a large enough dataset we would maybe see something that resembles this chart:

![](https://github.com/eitancohen77/NBA-Dataset-Analysis/assets/98838116/e3451ef5-ab9a-40a5-9119-7c1c10a5a3bd)
<br><br><br>

## Analysis 4: Injuries increasing due to Possessions

After seeing the number of possessions from the previous analysis, it led to me hypothesizing the idea that because possessions are increasing and the game pace is getting faster if there was a correlation with this and NBA players getting more injured. To do this I would need to pull another dataset which contains players that were injured and their reasoning. Luckily for me there was a very large dataset that dates back all the way to 1951 of every single injury that ever occurred. I am going to focus on the years 2013-2022 since these are the only years I have access to in my dataset. 

![](https://github.com/eitancohen77/NBA-Dataset-Analysis/assets/98838116/cf8d91fd-0021-42d9-831e-8cea45013c5d)

Relinquished and acquired are the ways in which the player got placed onto the injury list according to this dataset. I am only going to care about the Season attribute because with this I am able to group by and see how many players got injured since 2013.
To make this dataset fair, I would need to get rid of an outliers because due to covid in the year 2021 I am going to see a lot more people placed on the injury list. Players with covid does not matter to my hypothesis because I believe the increase in fast pacing of the game would affect injuries. 

![](https://github.com/eitancohen77/NBA-Dataset-Analysis/assets/98838116/f425abad-dd8b-4a3c-abf9-dd6d03196891)

500 results were removed to make it fair and now I can begin creating some visualization:

![](https://github.com/eitancohen77/NBA-Dataset-Analysis/assets/98838116/16d790b1-25c9-48f7-9f5f-daa2fd06312b)

<br>
Not a very pretty plot. Lets see what a linear regression test would say.

![](https://github.com/eitancohen77/NBA-Dataset-Analysis/assets/98838116/442b25c0-4969-4cb7-a9a4-9f95ac5bdc12)
- As  we can see, the pvalue is way too high meaning there is no significant evidence to suggest that the amount of injuries increases with the amount of possessions. 
<br><br><br>

## Analysis 5: Finding Statistically Closest Neighbors

I want to create a KNN algorithm which will take the average of the 21-22 season of all the players. This will be used as the training model. Then you can choose, or take a random player from the 2021-2022 season and see the 5 players who share the chosen player's stats. I am hoping from doing this you can see some common factors with players with the chosen player that goes beyond basketball. Say for example I choose ‘Kyrie Irving’ who is a player that is very reliant on his ability to dribble a basketball as my input player and the machine spits out Stephen Curry, a player who is also very reliant on dribbling a basketball.

For this calculation I tried normalizing the data, but when I imputed different players the same 5 players kept being outputted so I stuck with the normal calculation which came out surprisingly accurate. I also dropped games played because this stat just tells you how many games a certain player played which more correlates with how injury prone a player is not his playing basketball ability and so because of this I don't want this stat affecting valuable data.

![](https://github.com/eitancohen77/NBA-Dataset-Analysis/assets/98838116/aed80b93-7338-46fc-a78a-251a2d69cd5f)

![](https://github.com/eitancohen77/NBA-Dataset-Analysis/assets/98838116/cadce7d5-6cdf-4bc1-907c-b410fe740a2e)

![](https://github.com/eitancohen77/NBA-Dataset-Analysis/assets/98838116/bbe5fc9a-5b1e-435d-8432-b1ff8dff950c)

So if Jayson Tatum gets inputed into the model, Devin Booker, Jaylen Brown, Stephen Curry, Kevin Durant and Paul George get outputted. If you know basketball you know that this is a somewhat accurate description since these guys are all elite scorers which is who Jayson Tatum is. To really test this I wanted to see if it would work on centers as well, and not superstars so I used 2 subjects: Domantas Sabonis who is a center, and Chris paul who is a point guard.

![](https://github.com/eitancohen77/NBA-Dataset-Analysis/assets/98838116/2381b32a-d45a-400f-934c-151d776c41cc)
- These are all role playing centers just like Domantas Sabonis. None of these have insane superstar stats so this is a fairly accurate description.

![](https://github.com/eitancohen77/NBA-Dataset-Analysis/assets/98838116/d12e90f5-68b3-4862-9ba9-ae73fc2ead3b)
- It outputted a bunch of playmaking point guards which is a good description of who Chris Paul is.
<br><br><br>

## Analysis 6: Seeing average players with Clustering

I want to use K-Means Clustering on the 3 positions of the NBA (Guards, Forwards, and Centers). With this clustering I will be able to see the 2 types of players there are. Those who have a positive box plus minus (PLUS_MINUS) means they had a positive contribution to the scoreboard, and those who had a negative box plus minus means they have a negative contribution to them being on the court. To do this I am simply going to use the groupby function on the 3 positions which are Guards, Forwards, and Centers. 

To start, I ran a test to see what my K value should approximately be.

![](https://github.com/eitancohen77/NBA-Dataset-Analysis/assets/98838116/fb6c9cab-edda-424b-a135-5cc6f5086508)
![](https://github.com/eitancohen77/NBA-Dataset-Analysis/assets/98838116/70808ea7-19df-4f8e-9e4f-1ea7d1013a12)
![](https://github.com/eitancohen77/NBA-Dataset-Analysis/assets/98838116/ccb1dda0-1f94-4245-81e0-5b2bf36f6d2d)

It came out roughly the same with K value being around 3. So I used this value for my guards dataframe:

<br>
<h3>Guards</h3>

![](https://github.com/eitancohen77/NBA-Dataset-Analysis/assets/98838116/3796ea94-a59d-4864-af2b-8e7ebbae7e3c)

From PLUS_MINUS stat, it appears we got back 3 types of guard players. Those who have a  negative, positive and neutral impact on the court. 

- Surprisingly, Guards who had the highest points per game had a neutral impact on the court. I am assuming these are the players who have star power, but have no one else good on their team so they have more shots to take, but it does not mean they helped out. 
- I could not see a reason as to why the players with a positive PLUS_MINUS had their individual stats. Reason why is because anything guards did from cluster 1, cluster 2 did but better.
- One of the biggest reasons I see as to why guards from cluster 2 had bad impact is because they had the worst field goal percentage (FG_PCT) and 3 point percentage (FG3_PCT), as well as they had the fewest assists numbers which something guards are supposed to do is to facilitate the floor and pass the ball around so it makes sense that a failing guard would lack this stat.

The analysis found that guards with the highest points per game had a neutral impact on the court, suggesting their scoring may not significantly contribute to team success. Guards in cluster 2 had a negative impact, likely due to their poor field goal and three point percentages as well as few assists.

![](https://github.com/eitancohen77/NBA-Dataset-Analysis/assets/98838116/ddd84f17-33fc-420c-a790-582959f714c5)

<br>
<h3>Forwards</h3>

![](https://github.com/eitancohen77/NBA-Dataset-Analysis/assets/98838116/2a4fa9fe-75c7-47e8-9505-da2b616d917d)

Looks like we have the same types of players as the last position: positive, neutral and negative.

- Once again we see a relationship between players who get a lot of points and have it neutrally impacted when they are on the floor. Almost as if these players have empty stats which means they look good on paper, but on the court it doesn't really amount to winning.
- We can see the negative cluster has a field goal percentage as the lowest of the 3, whereas the neutral and positive clusters have around the same field goal percentage at around 50% each. Same stat applies to 3 point percentage.
  
Forwards who score a lot of points may not contribute significantly to winning, indicating “empty stats.” The negative cluster had the lowest field goal percentage, while the neutral and positive clusters had similar percentages around 50%. Shooting efficiency plays a crucial role on a player’s impact on the court.

<br>
<h3>Centers</h3>

![](https://github.com/eitancohen77/NBA-Dataset-Analysis/assets/98838116/db0ef174-6324-4567-b40e-7cc0eb5ef713)

Here is interesting because what is supposed to be a neutral PLUS_MINUS as the trend we have been seeing, the centers get a somewhat positive plus minus. This may tell us that when centers perform well and score a lot of points it may lead to a positive relationship on the court.
- As we can see from cluster 1, centers who don't rebound (REB) the ball well will have a negative effect on the court. This makes sense because a center’s primary job is to rebound the ball so there can either be a second chance to score or defend the rim and get back on offense. 
- Blocks (BLK) is also a stat in which the negative cluster lacks. 
- As the previous cluster groups showed it seems there is a trend with field goal percentage in that the worst a player shoots the ball, the worst plus minus that player would have

This analysis of clustering revealed that centers who struggle with rebounding and have a low field goal percentage tend to have a negative impact on the court. Effective rebounding and efficient scoring are crucial for centers to contribute positively to their team’s success.

Overall these clusters had a very similar message. Shooting the ball well does not automatically correlate with success on the basketball, but shooting the ball poorly will give you bad numbers when it comes to the plus minus stat.
<br><br>

# Conclusion
Throughout this project, I gained valuable insights and drew meaningful conclusions from analyzing an NBA dataset. The data exploration and statistical analyses provided a deeper understanding of player performance, changes over time and the impact of various factors on team success. From the analysis of statistical correlations between player stats, I observed interesting relationships between different variables, indicating that players who perform better in one area may not do so in the other. The examination of minutes played distribution revealed insights into the allocation of playing time, particularly in the playoffs. Analyzing the changes in the game over the past decade uncovered significant trends such as the steady increase in the three point shot which indicated a more perimeter-oriented style of play. While exploiting the relationship between possessions and injuries, the analysis did not find significant evidence to support direct correlation which suggests that an increase in possessions and pace may not inherently lead to a higher risk of injuries among NBA players. By using machine learning techniques, I used the K-Nearest Neighbors algorithm to identify similar players and group them based on their stats as well using K-Means Clustering to group certain players up to see which one had a great impact on the court. 
Overall this project deepened my knowledge of the NBA game and its  intricacies. The data-driven analysis offered valuable insights into player performance, the evolving nature of the game and the factors that contribute to team success. 

