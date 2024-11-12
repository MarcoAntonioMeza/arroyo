from django.urls import path
import arroyo.views as v

urlpatterns = [
     path('',v.index, name='ventas_index'),
     path('mes/',v.ventas_mes, name='ventas_mes'),
     path('productos/',v.productos, name='producto_detalle'),
     
     path('compras/',v.compras, name='compras'),
     path('compras/proveedor/',v.compras_proveedor, name='compras_proveedor'),
     
]