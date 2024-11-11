SELECT 
    cd.id,
    -- COMPRA 
    cd.compra_id AS id_compra,
    cm.`status`,
    
    -- PRODUCTO 
    p.id AS id_product, 
    p.nombre AS nombre_product,
    p.tipo AS tipo_product, 
    p.tipo_medida AS tipo_medida_product,
    p.costo AS costo_product, 

    -- COMPRA DETALLE
    cd.cantidad AS cantidad_compra, 
    cd.costo AS costo_compra,
    
    -- FECHA 
    FROM_UNIXTIME(cd.created_at) AS fecha,
    -- PROOVEDOR 
    pr.id AS id_proveedor,
    pr.nombre AS nombre_proveedor
    

FROM compra_detalle AS cd
INNER JOIN producto AS p ON p.id = cd.producto_id
INNER  JOIN compra AS cm ON cd.compra_id = cm.id 
LEFT JOIN proveedor pr ON cm.proveedor_id = pr.id
WHERE cm.`status` = 40
-- ORDER BY cd.id DESC;
