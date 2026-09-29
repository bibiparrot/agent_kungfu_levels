



make each level into a subdir and write a mariom python:

1. necessary configs are put in D:\Python\agent_kungfu_levels\src\agent_kungfu\levels,  with a yaml file under each subdir to describe config. 

2. baseline is a sql command in D:\Python\agent_kungfu_levels\src\agent_kungfu\levels\baseline\revenue.sql, please find config about D:\Python\agent_kungfu_levels\data\sakila.db  and sql revenue.sql in this yaml .

3. level1, design a workflow, 3 steps, read tables, merge, and  calculate

   - read tables category c, film_category fc, inventory i, rental  r and payment p into polars dataframe
   - merge, join into a big table  c.category_id = fc.category_id , fc.film_id = i.film_id, i.inventory_id = r.inventory_id,  r.rental_id = p.rental_id
   - calculate, groupby c.name, SUM(p.amount) into revenue and sort reverse. 
   - use a prompt to find top n categories by revenue and n will be a function call for llm with prompt use litellm. 

4. level2, design an analysis agents using openai-agents

   1. qa agent as a leader agent it calls table read agent and calculate agent and reply
   2. table read agent 
   3. calcualte agent

   

5. level3, design an analysis agents make an agent.md and codex python sdk to dinamically generate 2 sub agents read and calculate

6. level4, upgrade  to  omnigent-client and connect with codex, and load  D:\Python\agent_kungfu_levels\src\agent_kungfu\levels\level4\polars skills to  generate 2 sub agents read and calculate and evaluate.

5. level5, add ralph loop design and  omnigent-client and connect with codex, iterate over all tables and eventually find out revenue category and quit. 