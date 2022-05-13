 SELECT
   date_trunc('day', CAST(created_at AS TIMESTAMP)) AS interaction_date,
   json_extract_path_text(social_media, 'user_name') AS user_name,
   url,
   text 
FROM
   spc_raw_hi_loja_br.interactions i 
WHERE
   LOWER(user_name) <> 'loja' 
   AND json_extract_path_text(social_media, 'key') = 'instagram_user'     -- comentários instagram
   AND created_at >= '2022-01-01'
