#from lib.utils import *
from utils import * ###UNCOMMENT LINE FOR MAIN
countries = ['ARG']
starts = [730]
mids = [830]
stops = [831]
argfull = get_SIR_covid_datasets(countries, starts, mids, stops)

dictstarts= {}
dictstarts['URY'] = 660
dictstarts['USA'] = 550
dictstarts['ESP'] = 510
dictstarts['FRA'] = 530
dictstarts['BRA'] = 670
dictstarts['MEX'] = 550
dictstarts['PRY'] = 680
dictstarts['ITA'] = 530
dictstarts['ARG'] = 730
dictstarts['COL'] = 660
dictstarts['CHL'] = 690
dictstarts['PER'] = 660
dictstarts['DEU'] = 630
dictstarts['GBR'] = 500
dictstarts['IND'] = 710
dictstarts['JPN'] = 560

dictmids = {}
for key in dictstarts.keys():
    dictmids[key] = dictstarts[key] + 100

dictmids['DEU'] = dictstarts['DEU'] + 80
dictmids['GBR'] = dictstarts['GBR'] + 55
dictmids['ARG'] = dictstarts['ARG'] + 10

dictstops = {}
for key in dictstarts.keys():
    dictstops[key] = dictmids[key] + 1



countries = [key for key in dictstarts.keys()]
starts = [dictstarts[key] for key in dictstarts.keys()]
mids = [dictmids[key] for key in dictmids.keys()]
stops = [dictstops[key] for key in dictstops.keys()]

#countries = [ 'URY','USA', 'ESP', 'FRA','BRA','MEX','PRY', 'ITA', 'ARG', 'COL', 'CHL', 'PER', 'DEU', 'GBR', 'IND', 'JPN']


#countries = [ 'ITA']
#starts = [40, 50, 40, 40, 80, 90, 100, 40]
#starts = [660, 550, 510, 530, 670, 550,680, 530, 730, 660, 690, 660, 630, 500, 710,560]

#starts = [0]*len(countries)
#mids = [1200]*len(countries)
#stops = [1201]*len(countries)
#mids = [150, 160, 110, 110, 190, 200, 210, 150]
#stop = mid + 1.0*[1., 1., 1., 1., 1., 1., 1., 1.]
#stops = [151, 161, 111, 111, 191, 201, 211, 151]
datasets = get_SIR_covid_datasets(countries, starts, mids, stops)
epochs = 5000
logg_every_e = 200
eta_dual = 1e1000

estimator_vals = {}
estimator_vals['URY'] = {'epochs' : epochs,
                    'beta' : 1e-3/ datasets['URY']['population'],
                    'gamma' : 2e-3,
                    'T' : 1,
                    'population' : datasets['URY']['population'],
                    'eta' : 1e-4/ datasets['URY']['population'],
                    'eta_cent' : [0],
                    'eta_dual' : eta_dual,
                    'logging' : False,
                    'logg_every_e' : logg_every_e,
                    }
estimator_vals['USA'] = {'epochs' : epochs,
                    'beta' : 1e-10/ datasets['USA']['population'],
                    'gamma' : 2e-3,
                    'T' : 1,
                    'population' : datasets['USA']['population'],
                    'eta' : 1e-8/ datasets['USA']['population'],
                    'eta_cent' : [0],
                    'eta_dual' : eta_dual,
                    'logging' : False,
                    'logg_every_e' : logg_every_e,
                    }
estimator_vals['ESP'] = {'epochs' : epochs,
                    'beta' : 1e-3/ datasets['ESP']['population'],
                    'gamma' : 2e-3,
                    'T' : 1,
                    'population' : datasets['ESP']['population'],
                    'eta' : 1e-7/ datasets['ESP']['population'],
                    'eta_cent' : [0],
                    'eta_dual' : eta_dual,
                    'logging' : False,
                    'logg_every_e' : logg_every_e,
                    }
estimator_vals['FRA'] = {'epochs' : epochs,
                    'beta' : 1e-3/ datasets['FRA']['population'],
                    'gamma' : 2e-3,
                    'T' : 1,
                    'population' : datasets['FRA']['population'],
                    'eta' : 1e-4/ datasets['FRA']['population'],
                    'eta_cent' : [0],
                    'eta_dual' : eta_dual,
                    'logging' : False,
                    'logg_every_e' : logg_every_e,
                    }
estimator_vals['BRA'] = {'epochs' : epochs,
                    'beta' : 1e-3/ datasets['BRA']['population'],
                    'gamma' : 2e-3,
                    'T' : 1,
                    'population' : datasets['BRA']['population'],
                    'eta' : 1e-4/ datasets['BRA']['population'],
                    'eta_cent' : [0],
                    'eta_dual' : eta_dual,
                    'logging' : False,
                    'logg_every_e' : logg_every_e,
                    }
estimator_vals['MEX'] = {'epochs' : epochs,
                    'beta' : 1e-3/ datasets['MEX']['population'],
                    'gamma' : 2e-3,
                    'T' : 1,
                    'population' : datasets['MEX']['population'],
                    'eta' : 1e-4/ datasets['MEX']['population'],
                    'eta_cent' : [0],
                    'eta_dual' : eta_dual,
                    'logging' : False,
                    'logg_every_e' : logg_every_e,
                    }
estimator_vals['PRY'] = {'epochs' : epochs,
                    'beta' : 1e-2/ datasets['PRY']['population'],
                    'gamma' : 2e-3,
                    'T' : 1,
                    'population' : datasets['PRY']['population'],
                    'eta' : 1e-2/ datasets['PRY']['population'],
                    'eta_cent' : [0],
                    'eta_dual' : eta_dual,
                    'logging' : False,
                    'logg_every_e' : logg_every_e,
                    }
estimator_vals['ITA'] = {'epochs' : epochs,
                    'beta' : 1e-7/ datasets['ITA']['population'],
                    'gamma' : 2e-3,
                    'T' : 1,
                    'population' : datasets['ITA']['population'],
                    'eta' : 1e-6/ datasets['ITA']['population'],#LEO: changed from 1e-6
                    'eta_cent' : [0],
                    'eta_dual' : eta_dual,
                    'logging' : False,
                    'logg_every_e' : logg_every_e,
                    }
ls_of_eta = [estimator_vals[key]['eta'] for key in estimator_vals.keys()]
estimator_vals['centralized'] = {'epochs' : epochs,
                    'beta' : 1e-10,
                    'gamma' : 2e-3,
                    'T' : 1,
                    'population' : 1,
                    # 'eta' : 1e-15,
                    'eta' : 1e-18,
                    'eta_cent' : ls_of_eta,
                    'eta_dual' : eta_dual,
                    'logging' : False,
                    'logg_every_e' : logg_every_e,
                    }
estimator_vals['CLParametricSmall'] = {'epochs' : epochs,
                    'beta' : 1e-10,
                    'gamma' : 2e-3,
                    'T' : 1,
                    'population' : 1,
                    'eta' : 1e-15,
                    'eta_cent' : ls_of_eta,
                    # 'eta_dual' : 1e-3,
                    'eta_dual' : 1e0, # for smaller epsilons
                    # 'eta_dual' : 1e1,  # for larger epsilons
                    'logging' : True,
                    'logg_every_e' : logg_every_e,
                    }
estimator_vals['CLParametric'] = {'epochs' : epochs,
                    'beta' : 1e-10,
                    'gamma' : 2e-3,
                    'T' : 1,
                    'population' : 1,
                    'eta' : 1e-15,
                    'eta_cent' : ls_of_eta,
                    # 'eta_dual' : 1e-3,
                    # 'eta_dual' : 1e2, # for smaller epsilons
                    'eta_dual' : 1e1,  # for larger epsilons
                    'logging' : True,
                    'logg_every_e' : logg_every_e,
                    }
eta = 1e-18
estimator_vals['CLFunctional'] = {'epochs' :  logg_every_e*epochs,
                    'beta' : 1e-10,
                    'gamma' : 2e-3,
                    'T' : 1,
                    'population' : 1,
                    'eta' : eta,
                    'eta_cent' : [eta for key in estimator_vals.keys()],
                    'eta_dual' : 1e1,
                    'logging' : True,
                    'logg_every_e' : logg_every_e,
                    }
if __name__ == '__main__':
    print('countries:', countries)
    print('datasets:', datasets)
    markers = ['o', 's','o', 'D', '^', 'v', '<', '>', 'p', '*', 'h', 'H', '+', 'x', 'd', '|', '_']
    #plot infected data for each country
    #use different markers to distinguish countries
    i=0
    times = {}
    for country in countries:
        #if starts[i]>-400:
        times = np.arange(dictstarts[country], dictmids[country])
        plt.plot(times, datasets[country]['train']['I'], label=country, marker=markers[countries.index(country) % len(markers)])
        i += 1
        #df = pd.DataFrame({'times': times, 'I': datasets[country]['train']['I']})
        #df.to_csv(f'I_data_{country}.csv', index=False)

    plt.xlabel('Time')
    plt.ylabel('Infected Population')
    plt.title('Infected Population for each country')
    plt.legend()
    plt.show()
    #plot SIR data for each country