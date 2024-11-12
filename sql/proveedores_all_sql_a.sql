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