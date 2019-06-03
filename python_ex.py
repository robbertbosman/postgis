import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.image as mpimg

import matplotlib.ticker as ticker
from shapely.geometry import LineString, Point
from shapely.wkt import dumps, loads



from datetime import time
from datetime import datetime


from scipy.stats import chisquare
import numpy as np
import psycopg2


import itertools
from multiprocessing import Process, Lock
import multiprocessing
from threading import Thread


import os
import gdal
from gdalconst import GA_ReadOnly

import io
from io import StringIO


 





class bezetting_straat:

	def __init__(self, cur, schema, straat, dag):

		self.cur = cur

		plt.rc('text', usetex=True)
		plt.rc('font', family='STIXGeneral')
		plt.rc('text.latex', preamble=r'\usepackage{booktabs}\usepackage{colortbl}')

		self.schema = schema
		self.straat = straat
		self.dag = dag

		self.row_names = ['L', 'M', 'K']

		self.tabel = r''

		self.meetpunten = []

		
		self.cur.execute("""SELECT (substring(column_name::TEXT FROM 4 FOR 2) ||':'|| substring(column_name::TEXT FROM 7 FOR 2)) FROM information_schema.columns WHERE table_schema = '{0}' AND table_name = '{1}' AND ordinal_position > 2""".format(schema, dag[0]))
		rows = self.cur.fetchall()
		for row in rows:
		    self.meetpunten.append(row[0])

		self.runids = []
		#self.cur.execute("""select run_id from {0}.{0}_routes_zoekv6""".format(schema))
		self.cur.execute("""SELECT substring(table_name from char_length('{0}') + 5 for 2)::int FROM information_schema.tables WHERE table_schema = '{0}' AND table_name LIKE '%bron'""".format(schema))
		rows = self.cur.fetchall()
		for row in rows:
		    self.runids.append(row[0])

		self.x_hs = []
		self.x = []
		for uur in range(24):
			for kwartier in range(0, 60, 15):
				self.x.append(time(uur,kwartier).strftime("%H:%M"))
				self.x_hs.append((uur * 3600 + kwartier * 60) * 2)


		self.x_hs_tel = []
		self.x_index_tel = []
		for i, meetpunt in enumerate(self.meetpunten):
			self.x_hs_tel.append(self.x_hs[self.x.index(meetpunt)])
			self.x_index_tel.append(self.x.index(meetpunt))
	
		
		self.delta_x = [self.x_index_tel[0], self.x_index_tel[-1]]


		self.raam_x = self.x_hs[self.delta_x[0]:self.delta_x[1] + 1] 

		self.raam_xt = self.x[self.delta_x[0]:self.delta_x[1] + 1] 

		self.afbeelding = ''

		self.aantal_p = 0


	def latex_table2(self, rows):

	    table = r'\begin{tabular}{@{} l '

	    for c in range(len(self.meetpunten)):
	       
	        table += r'c'

	    table += r'@{}} \toprule {Parkeerder}&\multicolumn{%d}{c @{}} {Teltijdstippen}\\ \cmidrule(l){2-%d}' % (len(self.meetpunten), len(self.meetpunten) + 1)
	    table += r'&'


	    for c in range(len(self.meetpunten)-1):

	        table += '{%s}' % str(self.meetpunten[c])
	        table += r'&'

	    table += '{%s}' % str(self.meetpunten[-1])
	    table += r'\\  \midrule'


	    for r in range(len(self.row_names)):

	        table += r'{%s} &' % str(self.row_names[r])  
	       

	        for c in range(len(rows[r]) - 1):

	                table += r'{%s} &' % str(rows[r][c])

	        table += r'{%s} \\' % str(rows[r][-1])


	    table += r'\bottomrule \end{tabular}'

	    return table


	def latex_table(self, rows):


		table = r'\begin{tabular}{@{} l '

		for c in range(len(self.meetpunten)):
	       
			table += r'c'

		table += r'@{}} \toprule {Parkeerder}&\multicolumn{%d}{c @{}} {Telmeetpunten}\\ \cmidrule(l){2-%d}' % (len(self.meetpunten), len(self.meetpunten) + 1)
		table += r'&'


		for c in range(len(self.meetpunten)-1):

		    table += '{%s}' % str(self.meetpunten[c])
		    table += r'&'

		table += '{%s}' % str(self.meetpunten[-1])
		table += r'\\  \midrule'


		for r in range(len(self.row_names)):

		    table += r'{%s} &' % str(self.row_names[r])  
		   

		    for c in range(len(rows[r]) - 1):

		            table += r'{%s} &' % str(rows[r][c])

		    table += r'{%s} \\' % str(rows[r][-1])


		table += r'\midrule'

		table += r'{Totaal} &' 

		print(rows)

		for r in range(len(rows[-1]) - 1):

		    table += r'{%s} &' % str(rows[-1][r])  
		   

		table += r'{%s} \\' % str(rows[-1][-1])  

		table += r'\bottomrule \end{tabular}'

		return table

	def tex(self):

		self.tex = r'''
			\clearpage
			\subsection{%s}
			\begin{figure}[!htb]
			\centering
			\includegraphics[width=\textwidth,height=\textheight,keepaspectratio]{%s}
			\label{fig:digraph}
			\end{figure}
			''' % (self.straat.replace('ë', '{\\"e}').replace('é', "{\\'e}").replace('è', "{\\`e}"), self.afbeelding)

		

	def maak_plot(self):


		self.cur.execute("""select count(straatnaam) from {0}.{0}_run{1}_parks where straatnaam = '{2}'""".format(self.schema, '{:02}'.format(self.runids[0]), self.straat))
		self.aantal_p = self.cur.fetchone()[0]

		self.cur.execute("""select count(straatnaam) from {0}.park_telling_route where straatnaam = '{1}'""".format(self.schema, self.straat))
		self.aantal_p_tel = self.cur.fetchone()[0]

		print('Aantal p', self.straat, self.aantal_p_tel, self.aantal_p)

		


		# Table telling

		rij_naam = ['L', 'M', 'K']
		lmk = []
		for t in range(len(self.meetpunten)):

			lmk_q = """
			select 
			sum(case when states[{0}] like 'L%' then 1 else 0 end)  as l,
			sum(case when states[{0}] like 'M%' then 1 else 0 end)  as m,
			sum(case when states[{0}] like 'K%' then 1 else 0 end)  as k
			from {1}.park_telling_route
			where straatnaam = '{2}' 
			""".format(t + 1, self.schema, self.straat)

			print('lmk_q', lmk_q)

			self.cur.execute(lmk_q)

			rows = self.cur.fetchall()
			for row in rows:
				lmk.append( [ int(row[0]), int(row[1]), int(row[2])]  )

		lmk = list(map(list, zip(*lmk)))

		print('lmk', lmk)

		##########edit
		lmk_tel = lmk

		totaal = []

		for i in range( len(lmk_tel[0])   ):

			tot = 0

			for j in range(len(lmk_tel)):

				tot += lmk_tel[j][i]

			totaal.append(tot)

		lmk_tel.append(totaal)

		print('gem',lmk_tel)

		lmk_ = []
		for rij in lmk_tel:
			lmk_s = []
			for r in rij:

				lmk_s.append(  str(round(r, 1)) + ' (' + str(round(100 * r / self.aantal_p_tel)) + '\%)'       )
			lmk_.append(lmk_s)

		lmk = lmk_


		print("""select l, m, k, tot from {0}.telling_per_straat where straatnaam = '{1}'""".format(self.schema, self.straat))
		self.cur.execute("""select l, m, k, tot from {0}.telling_per_straat where straatnaam = '{1}'""".format(self.schema, self.straat))
		lmktot = self.cur.fetchall()[0]

		print(lmktot[0])


		# Table simulatie
		
		lmk_sim_tot = []

		for runid in self.runids:
			#print(runid)
			lmk_sim = []

			for t in range(len(self.meetpunten)):

				lmk_q = """
				select 
				sum(case when states[{0}] like 'L%' then 1 else 0 end)  as l,
				sum(case when states[{0}] like 'M%' then 1 else 0 end)  as m,
				sum(case when states[{0}] like 'K%' then 1 else 0 end)  as k
				from {1}.{1}_run{2}_parks
				where straatnaam = '{3}' 
				""".format(t + 1, self.schema, '{:02}'.format(runid), self.straat)

				self.cur.execute(lmk_q)

				rows = self.cur.fetchall()
				for row in rows:
					lmk_sim.append( [ row[0], row[1], row[2]]  )

			lmk_sim = list(map(list, zip(*lmk_sim)))

			lmk_sim_tot.append(lmk_sim)

	 

		lmk_sim_gem = []
		rn = []
		for i in range(len(rij_naam)):
			

			mp = []
			for j in range(len(self.meetpunten)):
				cel = 0
				for k in range(len(self.runids)):
					
					cel = cel + lmk_sim_tot[k][i][j] / len(self.runids)

				mp.append(cel)

			rn.append(mp)

		lmk_sim_gem.append(rn)

		print('gem',lmk_sim_gem)


		lmk_sim_gem = lmk_sim_gem[0]

		totaal = []

		for i in range( len(lmk_sim_gem[0])   ):

			tot = 0

			for j in range(len(lmk_sim_gem)):

				tot += lmk_sim_gem[j][i]

			totaal.append(tot)

		lmk_sim_gem.append(totaal)

		print('gem',lmk_sim_gem)

		lmk_sim = []
		for rij in lmk_sim_gem:
			lmk_s = []
			for r in rij:

				lmk_s.append(  str(round(r, 1)) + ' (' + str(round(100 * r / self.aantal_p)) + '\%)'       )
			lmk_sim.append(lmk_s)

		print(lmk_sim)


		#Polynoom bepalen van de meetpunten
		bezetting_telling = []

		for i, meetpunt in enumerate(self.meetpunten):
			

			sql = """
			select
			count(st)
			from 
				(
				select 
				states[{0}] as st
				from {1}.park_telling_route 
				where straatnaam = '{2}'
				) a
			where st != '0' 
			""".format(i + 1, self.schema, self.straat)

			#print(sql)


			self.cur.execute(sql)
			bezetting = self.cur.fetchone()[0]
			bezetting_telling.append(100 * bezetting / self.aantal_p_tel)

			print (i+1, meetpunt, 100 * bezetting / self.aantal_p_tel, self.straat)



		#matrix run_y met horizontaal 60 bezettinggraden, verticaal het aantal runs
		run_y = np.array([])

		for runid in self.runids:

			y = []

			for t in self.raam_x:
				

				sql = """
				select count(a.park_id) from 
				(
					select
					distinct
					d.park_id
					from {0}.{0}_run{1}_agents_dyn d
					join {0}.{0}_run{1}_agents_stat s
					on d.id_agent = s.id_agent
					join {0}.{0}_run{1}_parks p
					on d.park_id = p.id
					where 
						park_id > 0 								and 
						d.starttijd_parkeren + d.parkeerduur >= {2} and 
						d.starttijd_parkeren <= {2}					and
						p.straatnaam = '{3}'
				) a
				""".format(self.schema, '{:02}'.format(runid), t, self.straat)

				self.cur.execute(sql)

				y.append(100 * self.cur.fetchone()[0] / self.aantal_p)


			y_ = np.array(y)

			run_y = np.concatenate([run_y, y_])

		
		print(run_y.shape[0])

		aantal_kol = int(run_y.shape[0] / len(self.runids))
		
		y_assen = np.reshape(run_y, (-1, aantal_kol))  #-1 voor autmatisch het aantal rijen bepalen


		gem = []
		onder = []
		boven = []

		for i in range(aantal_kol):

			gem.append(np.mean(y_assen[:,i]))
			onder.append( np.mean(y_assen[:,i]) - np.std(y_assen[:,i]) )
			boven.append( np.mean(y_assen[:,i]) + np.std(y_assen[:,i]) )



		print(len(self.raam_xt), len(self.raam_x) )





		############################################Uitsnede

	
		fig = plt.figure(figsize=(7.5, 11.5))

		bezetting = plt.subplot2grid((15,10), (0,1), rowspan=3, colspan=6)   #rij 0 kol 0
		tabel1 = plt.subplot2grid((15,10), (3,0),  rowspan=2, colspan=6)
		tabel2 = plt.subplot2grid((15,10), (5,0),  rowspan=2, colspan=6)
		uitsnede = plt.subplot2grid((15,10), (8, 0), rowspan=6, colspan=10)


		runid = 1

		sql = """select type, box, st_astext(geom), rast from {0}.rb_closeup_straat_v2('{0}', {1}, '{2}', {3}, array[{4}, {5}]);""".format(self.schema, runid, self.straat, 0.25, 15, 7)
		print(sql)
		self.cur.execute(sql)
		rows = self.cur.fetchall()

		parks = []
		ways = []
		omg = []


		for row in rows:

			if row[0] == 'park':
				park = loads(row[2])
				parks.append(park)

			if row[0] == 'straat':
				way = loads(row[2])
				ways.append(way)
				#straat = loads(row[1])

			if row[0] == 'omgeving':
				omgeving = loads(row[2])
				#omg.append(omgeving)

			if row[0] == 'box':
				box = row[1]
				box_string = row[2]

				


		print('BOX', box_string)

		#Onderstaande in rb_closeup_straat_v2 geeft: rt_raster_from_wkb error
		#Boks cooridinaten kunnen eruit. 



		self.cur.execute("""
			SET postgis.gdal_enabled_drivers = 'ENABLE_ALL';
			SET postgis.enable_outdb_rasters = True;

		    select
			ST_AsGDALRaster(ST_Union(rast), 'GTiff') AS rast 
			--from public.arnhem_kb_3395 r
			--from public.pijp800_geov3 r
			from public.zeist_3395_4 r
			where st_intersects(r.rast , ST_AsEWKB(ST_GeomFromText('{0}', 3395))); 
			""".format(box_string))

		f = open("/tmp/aap.tiff", "wb")
		f.write(self.cur.fetchone()[0])
		f.close()




		#coordinaten uit de tiff halen. Blijkt alleen op deze manier te werken, anders liggen de straatlijen niet ok
		data = gdal.Open('/tmp/aap.tiff', GA_ReadOnly)
		geoTransform = data.GetGeoTransform()
		minx = geoTransform[0]
		maxy = geoTransform[3]
		maxx = minx + geoTransform[1] * data.RasterXSize
		miny = maxy + geoTransform[5] * data.RasterYSize
		print('GDAL BOX',  [minx, miny, maxx, maxy])
		data = None



		img = plt.imread('/tmp/aap.tiff')
		os.remove("/tmp/aap.tiff")
		# uitsnede.imshow(img, extent=[box[0], box[1], box[2], box[3]])
		uitsnede.imshow(img, extent=(minx, maxx, miny, maxy))

		for park in parks:
		    x_p, y_p = park.xy
		    uitsnede.plot(x_p, y_p, 'o', color=(0, 0, 0), zorder=1)

		for w in ways:
			x_w, y_w = w.xy
			uitsnede.plot(x_w, y_w, color=(1, 0, 0), alpha=0.7, linewidth=2.5, solid_capstyle='round', zorder=2)


		
		boks = [0.15,0,0.7,0.7]



		print('lmk', lmk)
		print('rij_naam', rij_naam)
		print('meetpunten', self.meetpunten)




		##################Tabellen
		knijp = -0.12
		dy_tit = 0.6
		dy_tot = -0.7

		font_s = 8


		jj = self.latex_table(lmk)
		tabel1.text(0, 0.5 - knijp + dy_tot, jj, size=font_s, va="center" ) 
		tabel1.text(0, 0.5 + dy_tit - knijp + dy_tot, 'Tabel 1: Samensenstelling parkeerders uit tellingen.', size=font_s, va="center")
		tabel1.axis('off')


		kk = self.latex_table(lmk_sim)
		tabel2.text(0, .5 + knijp + dy_tot, kk, size=font_s, va="center")  #.5,.5 wil zeggen dat het midden van de tabel in het midden van de subplot ligt
		tabel2.text(0, 0.5 + dy_tit + knijp + dy_tot, 'Tabel 2: Gemiddelde samensenstelling parkeerders ({0} simulaties).'.format(len(self.runids)), size=font_s, va="center")
		tabel2.axis('off')

		##################Tabellen



		bezetting.set_axisbelow(True)
		bezetting.grid(False, color='w', linestyle='-', linewidth=1, alpha=0.75)
		bezetting.set_facecolor('lightgrey')


		bezetting.set_xlabel('tijd op de dag')
		bezetting.set_ylabel('bezettingsgraad (\%)')
		bezetting.set_ylim([0, 100])

		bezetting.plot(self.raam_xt, gem, color='black')

		bezetting.fill_between(self.raam_xt, onder, boven, color='gray', alpha=0.2)

		bezetting.scatter(self.meetpunten, bezetting_telling, marker='+', s=400, color='black')   #Deze als laatste, geen idee waarom

		plt.setp(bezetting, xticks=self.meetpunten, xticklabels=self.meetpunten)


		uitsnede.set_aspect(1)
		#uitsnede.set_title('Uitsnede {0}'.format(self.straat))
		#uitsnede.set_xlabel("""L:{0}, M:{1}, K:{2}, totaal aantal parkeerders: {3}""".format(lmktot[0], lmktot[1], lmktot[2], lmktot[3]))
		

		uitsnede.tick_params(axis='both', which='both', length=0)
		uitsnede.xaxis.set_ticklabels([])
		uitsnede.yaxis.set_ticklabels([])





		filename = """/{0}_{1}_{2}_{3}_{4}.pdf""".format(self.schema, self.straat, self.runids[0], self.runids[-1], self.aantal_p)

		dir = """./pica/{0}/{1}_{2}""".format(self.schema, self.runids[0], self.runids[-1])

		pad = (dir + filename).replace(' ', '_')

		self.afbeelding = pad

		if not os.path.exists(dir):
			os.makedirs(dir)

		plt.savefig(pad, format='pdf', dpi=300, bbox_inches='tight')

		plt.close()
		self.tex()


		# plt.show()


