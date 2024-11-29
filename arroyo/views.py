from django.utils import timezone
from datetime import timedelta
from datetime import date
from django.shortcuts import render
from process_data.process import *




def index(request):
    data = vetas_totales()
    return render(request, 'ventas/index.html',data)

def ventas_mes(request):
    year = int(request.GET.get('year',date.today().year))
    data = ventas_mes_df(year)
    return render(request, 'ventas/mes.html',data)



def productos(request):
    data = venta_detalle_producto()
    return render(request, 'productos/index.html',data)

def producto_compra_ventas(request):
    producto = request.GET.get('producto')
    data = None #producto_compra_ventas(producto)
    return render(request, 'productos/producto.html',data)

#================================================================
#                                   COMPRAS 
#================================================================
def compras(request):
    year = int(request.GET.get('year',date.today().year))
    data = compras_credito_abonos(year)
    return render(request, 'compras/index.html',data)



def compras_proveedor(request):
    COUNT_PROV = 200
    df_pr = consulta_sql(PROOVEDORES_LIST)
    df_pr['proveedor'] = df_pr['proveedor'].str.upper().str.strip().str.replace('_', ' ')
    lista_pro = df_pr['proveedor'].unique()[:COUNT_PROV].tolist()
    select_pro = request.GET.get('proveedor',lista_pro[0])
   
    
    fecha_actual = timezone.now().date()
    fecha_inicio_default = fecha_actual - timedelta(days=30)
    
    fecha_fin = request.GET.get('fecha_fin',fecha_actual.strftime('%Y-%m-%d'))
    fecha_inicio = request.GET.get('fecha_inicio',fecha_inicio_default.strftime('%Y-%m-%d'))
    
    
    
    compras_pro = compras_proveedor_pro(select_pro,fecha_inicio,fecha_fin)
    
    
    
    
    data = {
        'provedores_list':{
            'lista':lista_pro,
            'select':select_pro
        },
        'compras':compras_pro,
        
        'fecha_fin': fecha_fin,
        'fecha_inicio': fecha_inicio,
    }
    #print(data)
    
    return render(request, 'compras/proveedor.html',data)