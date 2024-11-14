VENTAS = """
SELECT
v.id AS venta_id, v.total AS venta_total,  v.tipo AS venta_tipo,
 v.cliente_id AS venta_cliente , FROM_UNIXTIME(v.created_at) AS fecha
FROM venta AS v;
"""

CREDITO = """
SELECT c.id AS credito_id, c.venta_id, c.cliente_id, c.monto,
c.`status`, c.tipo,
FROM_UNIXTIME(c.created_at) AS fecha
FROM credito AS c
WHERE c.tipo = 10 ;
"""

ABONO = """
SELECT
ca.id AS credito_abono_id, ca.credito_id, ca.cantidad, ca.`status`, ca.token_pay,
c.venta_id, c.cliente_id,
FROM_UNIXTIME(ca.created_at) AS fecha
FROM credito_abono AS ca
JOIN credito AS c ON c.id = ca.credito_id
WHERE c.tipo = 10;
"""



VENTA_DETALLE = """
SELECT
    vd.id,
    -- c.nombre AS cliente,
    -- c.id AS cliente_id,
    -- vd.venta_id,
    -- vd.producto_id,
    p.nombre AS producto,
    p.costo,
    -- p.precio_publico,
    -- p.precio_mayoreo,
    -- p.precio_menudeo,
     vd.cantidad,
    -- vd.precio_venta,
    -- vd.created_by,
    -- u.nombre AS creado_por,
    p.tipo_medida,
    -- Ajuste de cantidad dependiendo del tipo de medida
    CASE
        WHEN p.tipo_medida = 20 THEN 'Kilos'
        WHEN p.tipo_medida = 10 THEN 'Piezas'
        ELSE '--'
    END AS unidad_medida,
   --  pertenece.id AS pertenece_id,
    -- pertenece.nombre AS pertenece,
    -- suc.id AS ruta_asignada_id,
    -- suc.nombre AS ruta_asignada,
    -- vd.created_at,
    DATE(FROM_UNIXTIME(vd.created_at)) AS fecha,  -- Solo la fecha
    TIME(FROM_UNIXTIME(vd.created_at)) AS hora
FROM
    venta_detalle AS vd
INNER JOIN
    producto AS p ON p.id = vd.producto_id
INNER JOIN
    user AS u ON u.id = vd.created_by
LEFT JOIN
    venta AS v ON vd.venta_id = v.id  -- Cambiado de vd.id a vd.venta_id para corregir el JOIN
LEFT JOIN
    cliente AS c ON c.id = v.cliente_id
LEFT JOIN
    sucursal AS suc ON v.ruta_sucursal_id = suc.id
LEFT JOIN
    sucursal AS pertenece ON v.sucursal_id = pertenece.id
"""



""""
====================================================
    COMPRAS
====================================================
"""
COMPRAS_PROVEDOR_C = """
SELECT 
-- COMPRA
cm.id AS id,
-- cm.lat, cm.lng,
cm.total AS total_compra,
cm.`status`,

-- PROVEEDOR
pr.nombre AS proveedor,
pr.id AS proveedor_id,


-- FECHA 
FROM_UNIXTIME(cm.created_at) AS fecha,
FROM_UNIXTIME(cm.fecha_salida) AS fecha_salida
FROM compra AS cm   
JOIN proveedor AS pr ON pr.id = cm.proveedor_id
WHERE cm.`status` = 40
ORDER BY id DESC ;


"""

CREDITO_PROVEDOR = """
SELECT
 -- CREDITO 
 c.id AS credito_id,
 round(c.monto,2) AS monto_credito,
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
WHERE c.`status` != 20

"""

ABONO_PROVEDOR = """
SELECT
ca.id AS credito_abono_id, 
ca.credito_id,
 ca.cantidad,
 --ca.`status`,
 -- ca.token_pay,
 c.compra_id,
 -- cm.proveedor_id,
-- c.tipo AS tipo,
upper(pr.nombre) AS proveedor,
FROM_UNIXTIME(ca.created_at) AS fecha
FROM credito_abono AS ca
JOIN credito AS c ON c.id = ca.credito_id
JOIN compra AS cm ON cm.id = c.compra_id
JOIN proveedor AS pr ON pr.id = cm.proveedor_id
WHERE ca.`status` = 10
 -- WHERE c.tipo = 20;
"""





PROOVEDORES_LIST = """
SELECT 
pr.id,
UPPER(pr.nombre) AS proveedor,
COUNT(cm.proveedor_id) AS total_compras
FROM compra AS cm
JOIN proveedor AS pr ON pr.id = cm.proveedor_id
WHERE cm.`status` = 40
GROUP BY cm.proveedor_id
ORDER BY total_compras DESC 
"""


COMPRAS_PROVEDOR = """
SELECT 
    cd.id,
    -- COMPRA 
    cd.compra_id AS id_compra,
    -- cm.`status`,
    
    
    -- PRODUCTO 
    p.id AS id_product, 
    p.nombre AS nombre_product,
    -- p.tipo AS tipo_product, 
    p.tipo_medida AS tipo_medida_product,
    -- p.costo AS costo_product, 

    -- COMPRA DETALLE
    cd.cantidad AS cantidad_compra, 
    cd.costo AS costo_compra,
    cm.total,
    
    -- FECHA 
    FROM_UNIXTIME(cd.created_at) AS fecha,
    -- PROOVEDOR 
    pr.id AS id_proveedor,
    UPPER(pr.nombre) AS proveedor
    

FROM compra_detalle AS cd
INNER JOIN producto AS p ON p.id = cd.producto_id
INNER  JOIN compra AS cm ON cd.compra_id = cm.id 
LEFT JOIN proveedor pr ON cm.proveedor_id = pr.id
WHERE cm.`status` = 40
 ORDER BY cd.id DESC;
"""



PROOVEDORES_LIST = """
SELECT 
pr.id,
UPPER(pr.nombre) AS proveedor,
-- COUNT(cm.proveedor_id) AS total_compras,
ROUND(SUM(cm.total),2) AS total_money
FROM compra AS cm
JOIN proveedor AS pr ON pr.id = cm.proveedor_id
WHERE cm.`status` = 40
GROUP BY cm.proveedor_id
ORDER BY total_money DESC 
"""


TOTALES_PROVEEDORES = """
SELECT
    pr.id AS id_proveedor,
    UPPER(pr.nombre) AS proveedor,
    COUNT(cm.proveedor_id) AS total_compras,
    ROUND(SUM(cm.total), 2) AS sum_total_compras,
    COALESCE(credito_data.monto_credito, 0) AS monto_credito,
    -- Si no hay crédito, muestra 0
    COALESCE(abono_data.total_abono, 0) AS monto_abono,
    COALESCE(credito_data.monto_credito, 0) - COALESCE(abono_data.total_abono, 0) AS por_pagar
FROM
    compra AS cm
    JOIN proveedor AS pr ON pr.id = cm.proveedor_id
    LEFT JOIN (
        SELECT
            pr.id AS id_proveedor,
            ROUND(SUM(c.monto), 2) AS monto_credito
        FROM
            credito AS c
            INNER JOIN compra AS cm ON cm.id = c.compra_id
            INNER JOIN proveedor AS pr ON pr.id = cm.proveedor_id
        WHERE
            c.`status` != 20
        GROUP BY
            pr.id
    ) AS credito_data ON pr.id = credito_data.id_proveedor
    LEFT JOIN (
        SELECT
            cm.proveedor_id AS pr_id,
            ROUND(SUM(ca.cantidad), 2) AS total_abono
        FROM
            credito_abono AS ca
            JOIN credito AS c ON c.id = ca.credito_id
            JOIN compra AS cm ON cm.id = c.compra_id
            JOIN proveedor AS pr ON pr.id = cm.proveedor_id
        WHERE
            ca.`status` = 10
        GROUP BY
            pr.id
    ) AS abono_data ON pr.id = abono_data.pr_id
WHERE
    cm.`status` = 40
GROUP BY
    cm.proveedor_id
ORDER BY
    sum_total_compras DESC;

"""