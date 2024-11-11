SELECT
ca.id AS credito_abono_id, ca.credito_id,
 ca.cantidad, ca.`status`, ca.token_pay,
 c.compra_id,
 cm.proveedor_id,
c.tipo AS tipo,
pr.nombre AS proovedor,
FROM_UNIXTIME(ca.created_at) AS fecha
FROM credito_abono AS ca
JOIN credito AS c ON c.id = ca.credito_id
JOIN compra AS cm ON cm.id = c.compra_id
JOIN proveedor AS pr ON pr.id = cm.proveedor_id
 -- WHERE c.tipo = 20;