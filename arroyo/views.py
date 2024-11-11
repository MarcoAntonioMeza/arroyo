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
    