SELECT
 -- CREDITO 
 c.id AS credito_id,
 c.monto AS monto_credito,
 -- COMPRA
 c.compra_id,
 cm.total AS moto_compra, 
 -- PROOVEDOR 
 pr.nombre AS proveedor,
 pr.id AS id_proveedor,
 c.`status`, c.tipo,

-- FECHA 
FROM_UNIXTIME(c.created_at) AS fecha

FROM credito AS c
INNER JOIN compra AS cm ON cm.id = c.compra_id
INNER  JOIN proveedor AS pr ON pr.id = cm.proveedor_id
-- WHERE c.id = 9598
