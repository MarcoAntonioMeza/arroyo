SELECT 
pr.id,
UPPER(pr.nombre) AS proveedor,
COUNT(cm.proveedor_id) AS total_compras
FROM compra AS cm
JOIN proveedor AS pr ON pr.id = cm.proveedor_id
WHERE cm.`status` = 40
GROUP BY cm.proveedor_id
ORDER BY total_compras DESC 