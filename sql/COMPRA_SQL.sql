SELECT 
-- COMPRA
cm.id AS id,
cm.lat, cm.lng, cm.total AS total_compra,
cm.`status`,

-- PROVVEEDOR
pr.nombre AS proveedor,
pr.id AS proveedor_id,


-- FECHA 
FROM_UNIXTIME(cm.created_at) AS fecha,
FROM_UNIXTIME(cm.fecha_salida) AS fecha_salida
FROM compra AS cm   
JOIN proveedor AS pr ON pr.id = cm.proveedor_id
WHERE cm.`status` = 40
ORDER BY id DESC ;
