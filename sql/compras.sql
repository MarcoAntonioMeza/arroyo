    SELECT 

        c.id AS id,
        c.sucursal_id AS sucursal_id,
        s.nombre AS sucursal,
        c.is_especial AS is_especial,
        c.proveedor_id AS proveedor_id,
        p.nombre AS proveedor,
        
        -- Suma de cantidades en compra_detalle
        (
            SELECT SUM(cd.cantidad)
            FROM compra_detalle cd
            WHERE cd.compra_id = c.id
        ) AS count_detalle,
        
        -- Suma de cantidades en operacion_detalle
        (
            SELECT SUM(od.cantidad)
            FROM operacion_detalle od
            JOIN operacion o ON od.operacion_id = o.id
            WHERE o.compra_id = c.id
        ) AS count_entrada_detalle,
        
        c.tiempo_recorrido AS tiempo_recorrido,
        c.fecha_salida AS fecha_salida,
        c.total AS total,
        c.status AS status,
        c.created_by AS created_by,
        c.created_at AS created_at,
        CONCAT_WS(' ', cr.nombre, cr.apellidos) AS created_by_user,
        c.updated_by AS updated_by,
        c.updated_at AS updated_at,
        CONCAT_WS(' ', up.nombre, up.apellidos) AS updated_by_user

    FROM compra c
    LEFT JOIN sucursal s ON c.sucursal_id = s.id
    LEFT JOIN proveedor p ON c.proveedor_id = p.id
    LEFT JOIN user cr ON c.created_by = cr.id
    LEFT JOIN user up ON c.updated_by = up.id;
