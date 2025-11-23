import yfinance as yf
import pandas as pd
import concurrent.futures
import sys
import time

# Fix encoding for Windows console
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

def get_bse_stock_codes():
    """
    Get a curated list of major BSE stock codes.
    These are primarily BSE 500 constituents and other actively traded stocks.
    """
    print("Loading BSE stock codes...")

    # Major BSE stocks - curated list of BSE 500 and actively traded companies
    # These codes are from BSE Sensex, BSE 100, BSE 200, and BSE 500
    bse_codes = [
        # BSE Sensex stocks
        '500325', '532540', '500180', '500209', '500696', '532174', '500247', '500112',
        '532454', '500875', '532215', '500510', '500820', '532500', '532281', '500034',
        '507685', '500114', '532538', '500790', '524715', '500312', '532555', '532898',
        '500570', '500470', '532921', '500520', '500228', '532187', '532755', '532978',
        '500124', '500300', '533278', '500087',

        # BSE 100 additional stocks
        '500425', '500830', '532712', '500490', '500290', '500440', '500182', '500387',
        '500410', '500547', '532541', '532868', '500304', '500790', '500285', '500575',
        '532155', '500780', '500878', '500790', '532762', '532712', '500041', '500103',

        # BSE 200 range
        '500010', '500013', '500020', '500026', '500027', '500028', '500033', '500038',
        '500039', '500042', '500043', '500044', '500049', '500051', '500053', '500055',
        '500056', '500059', '500060', '500067', '500069', '500071', '500072', '500074',
        '500076', '500077', '500078', '500080', '500082', '500084', '500085', '500086',
        '500088', '500089', '500092', '500093', '500094', '500095', '500096', '500097',
        '500098', '500099', '500101', '500102', '500104', '500105', '500106', '500107',
        '500108', '500109', '500110', '500111', '500113', '500115', '500116', '500117',
        '500119', '500121', '500125', '500126', '500127', '500128', '500131', '500132',
        '500134', '500135', '500136', '500137', '500139', '500141', '500142', '500143',
        '500144', '500145', '500146', '500147', '500148', '500149', '500150', '500151',
        '500152', '500153', '500154', '500155', '500156', '500159', '500160', '500161',
        '500163', '500164', '500165', '500166', '500167', '500168', '500169', '500170',
        '500171', '500173', '500174', '500175', '500176', '500177', '500178', '500179',
        '500181', '500183', '500185', '500186', '500187', '500188', '500189', '500191',
        '500192', '500193', '500195', '500196', '500197', '500198', '500199', '500200',
        '500201', '500202', '500203', '500205', '500206', '500207', '500208', '500210',
        '500211', '500212', '500213', '500214', '500215', '500216', '500217', '500218',
        '500219', '500220', '500222', '500224', '500225', '500226', '500227', '500230',

        # 50xxxx range (selective major companies)
        '502000', '502008', '502010', '502040', '502144', '502180', '502186', '502195',
        '502230', '502255', '502260', '502280', '502300', '502355', '502391', '502400',

        # 505xxx range (many established companies)
        '505185', '505200', '505204', '505218', '505283', '505284', '505330', '505350',
        '505371', '505400', '505410', '505426', '505428', '505450', '505500', '505506',
        '505509', '505520', '505533', '505537', '505550', '505560', '505571', '505592',
        '505600', '505606', '505610', '505630', '505660', '505690', '505700', '505710',
        '505714', '505715', '505720', '505730', '505750', '505760', '505770', '505780',
        '505790', '505800', '505810', '505820', '505826', '505830', '505840', '505857',

        # 507xxx range
        '507253', '507255', '507260', '507330', '507424', '507425', '507470', '507480',
        '507500', '507506', '507508', '507509', '507517', '507526', '507527', '507550',
        '507570', '507580', '507600', '507640', '507650', '507670', '507680', '507683',
        '507686', '507688', '507690', '507700', '507710', '507720', '507726', '507730',
        '507770', '507775', '507785', '507795', '507800', '507801', '507806', '507808',

        # 532xxx range - major modern listings
        '532000', '532001', '532003', '532004', '532005', '532007', '532008', '532009',
        '532011', '532013', '532015', '532017', '532020', '532027', '532029', '532031',
        '532032', '532035', '532037', '532040', '532045', '532046', '532047', '532048',
        '532053', '532054', '532055', '532057', '532058', '532059', '532060', '532061',
        '532062', '532063', '532064', '532065', '532066', '532067', '532069', '532070',
        '532071', '532073', '532074', '532075', '532076', '532077', '532078', '532079',
        '532080', '532081', '532082', '532083', '532085', '532086', '532087', '532088',
        '532089', '532090', '532091', '532092', '532093', '532094', '532095', '532097',
        '532098', '532100', '532102', '532103', '532104', '532105', '532106', '532107',
        '532108', '532109', '532110', '532111', '532114', '532115', '532116', '532117',
        '532118', '532119', '532120', '532121', '532122', '532123', '532124', '532125',
        '532127', '532128', '532129', '532130', '532131', '532132', '532133', '532134',
        '532135', '532136', '532137', '532138', '532139', '532140', '532141', '532142',
        '532143', '532144', '532145', '532146', '532147', '532148', '532149', '532150',
        '532151', '532152', '532153', '532154', '532156', '532157', '532158', '532159',
        '532160', '532161', '532162', '532163', '532164', '532165', '532166', '532167',
        '532168', '532169', '532170', '532171', '532172', '532173', '532175', '532176',
        '532177', '532178', '532179', '532180', '532181', '532182', '532183', '532184',
        '532185', '532186', '532188', '532189', '532190', '532191', '532192', '532193',
        '532194', '532195', '532196', '532197', '532198', '532199', '532200', '532201',
        '532202', '532203', '532204', '532205', '532206', '532207', '532208', '532209',
        '532210', '532211', '532212', '532213', '532214', '532216', '532217', '532218',
        '532219', '532220', '532221', '532222', '532223', '532224', '532225', '532226',
        '532227', '532228', '532229', '532230', '532231', '532232', '532233', '532234',
        '532235', '532236', '532237', '532238', '532239', '532240', '532241', '532242',
        '532243', '532244', '532245', '532246', '532247', '532248', '532249', '532250',
        '532251', '532252', '532253', '532254', '532255', '532256', '532257', '532258',
        '532259', '532260', '532261', '532262', '532263', '532264', '532265', '532266',
        '532267', '532268', '532269', '532270', '532271', '532272', '532273', '532274',
        '532275', '532276', '532277', '532278', '532279', '532280', '532282', '532283',
        '532284', '532285', '532286', '532287', '532288', '532289', '532290', '532291',
        '532292', '532293', '532294', '532295', '532296', '532297', '532298', '532299',

        # Continue with selective 53xxxx range (newer listings)
        '533030', '533040', '533050', '533080', '533096', '533100', '533106', '533120',
        '533122', '533135', '533140', '533150', '533151', '533155', '533160', '533162',
        '533163', '533174', '533183', '533189', '533193', '533200', '533206', '533208',
        '533218', '533228', '533229', '533230', '533234', '533248', '533254', '533260',
        '533261', '533266', '533271', '533273', '533274', '533275', '533276', '533277',
        '533280', '533283', '533284', '533285', '533286', '533287', '533288', '533289',
        '533290', '533291', '533292', '533293', '533294', '533295', '533296', '533300',
        '533308', '533313', '533317', '533318', '533319', '533320', '533323', '533330',
        '533333', '533339', '533344', '533347', '533350', '533351', '533355', '533356',
        '533358', '533359', '533364', '533365', '533366', '533368', '533369', '533373',
        '533374', '533376', '533377', '533378', '533380', '533381', '533382', '533384',

        # 534xxx-54xxxx range (recent IPOs and new listings)
        '534309', '534312', '534325', '534330', '534338', '534339', '534345', '534350',
        '534356', '534364', '534374', '534378', '534380', '534387', '534395', '534399',
        '534400', '534425', '534437', '534440', '534444', '534450', '534455', '534460',
        '534477', '534478', '534488', '534495', '534496', '534497', '534500', '534506',

        # 5390xx-543xxx range
        '539123', '539136', '539144', '539145', '539150', '539200', '539254', '539268',
        '539277', '539301', '539333', '539336', '539345', '539351', '539366', '539367',
        '539404', '539415', '539416', '539417', '539418', '539437', '539438', '539448',
        '539463', '539477', '539488', '539501', '539507', '539509', '539523', '539540',
        '539543', '539556', '539558', '539559', '539578', '539599', '539611', '539650',
        '539660', '539678', '539680', '539683', '539685', '539687', '539692', '539693',
        '539712', '539721', '539735', '539740', '539750', '539762', '539768', '539777',
        '539799', '539802', '539806', '539824', '539840', '539844', '539845', '539876',
        '539877', '539889', '539893', '539902', '539936', '539937', '539938', '539940',
        '539942', '539943', '539945', '539957', '539971', '539976', '539986', '539995',
        '540025', '540030', '540064', '540065', '540075', '540087', '540119', '540124',
        '540133', '540136', '540153', '540173', '540179', '540195', '540200', '540210',
        '540222', '540268', '540277', '540300', '540315', '540376', '540399', '540400',
        '540425', '540450', '540475', '540500', '540525', '540575', '540611', '540613',
        '540644', '540678', '540679', '540691', '540699', '540700', '540710', '540716',
        '540719', '540744', '540762', '540777', '540797', '540798', '540799', '540800',
        '540823', '540828', '540830', '540868', '540902', '540903', '540964', '540999',
        '541001', '541108', '541153', '541154', '541179', '541195', '541198', '541228',
        '541230', '541246', '541259', '541268', '541277', '541300', '541336', '541344',
        '541450', '541500', '541540', '541597', '541600', '541669', '541700', '541770',
        '541800', '541900', '541930', '542066', '542229', '542230', '542233', '542234',
        '542239', '542241', '542244', '542283', '542365', '542389', '542400', '542415',
        '542437', '542451', '542477', '542503', '542520', '542539', '542540', '542546',
        '542593', '542602', '542651', '542663', '542665', '542667', '542670', '542675',
        '542704', '542710', '542726', '542735', '542750', '542778', '542830', '542851',
        '542902', '542904', '542905', '542909', '542912', '542920', '542923', '542925',
        '542935', '542937', '542942', '542943', '542946', '542960', '542965', '542966',
        '542979', '542994', '543220', '543228', '543257', '543257', '543272', '543299',
        '543310', '543321', '543348', '543397', '543430', '543472', '543490', '543513',
        '543518', '543520', '543526', '543529', '543544', '543556', '543572', '543576',
        '543620', '543654', '543657', '543665', '543693', '543712', '543713', '543720',
        '543730', '543874', '543885', '543907', '543908', '543909', '543913', '543915',
        '543931', '543949', '543950', '543951', '543960', '543971', '543996',
    ]

    bse_tickers = [code + '.BO' for code in bse_codes]
    print(f"Loaded {len(bse_tickers)} BSE ticker codes")
    return bse_tickers

def get_nse_stock_list():
    """
    Get comprehensive list of NSE stocks from the previous script.
    """
    print("Loading NSE stock symbols...")

    # Reusing NSE list from previous script
    known_tickers = [
        'RELIANCE', 'TCS', 'HDFCBANK', 'INFY', 'HINDUNILVR', 'ICICIBANK', 'KOTAKBANK',
        'SBIN', 'BHARTIARTL', 'ITC', 'AXISBANK', 'LT', 'ASIANPAINT', 'MARUTI', 'HCLTECH',
        'BAJFINANCE', 'WIPRO', 'TITAN', 'ULTRACEMCO', 'NESTLEIND', 'SUNPHARMA', 'ONGC',
        'NTPC', 'POWERGRID', 'TATAMOTORS', 'TATASTEEL', 'ADANIPORTS', 'M&M', 'JSWSTEEL',
        'INDUSINDBK', 'TECHM', 'BAJAJFINSV', 'DRREDDY', 'GRASIM', 'COALINDIA', 'CIPLA',
        'BRITANNIA', 'DIVISLAB', 'EICHERMOT', 'SHREECEM', 'HINDALCO', 'HEROMOTOCO',
        'APOLLOHOSP', 'BPCL', 'UPL', 'ADANIENT', 'ADANIGREEN', 'TATACONSUM', 'HDFCLIFE',
        'SBILIFE', 'BAJAJ-AUTO', 'PIDILITIND', 'BERGEPAINT', 'HAVELLS', 'GODREJCP',
        'DABUR', 'MARICO', 'COLPAL', 'VBL', 'TATAPOWER', 'LICHSGFIN', 'BANKBARODA',
        'PNB', 'CANBK', 'UNIONBANK', 'IDFCFIRSTB', 'FEDERALBNK', 'PFC', 'RECLTD',
        'NBCC', 'DLF', 'GODREJPROP', 'OBEROIRLTY', 'PRESTIGE', 'PHOENIXLTD', 'ACC',
        'AMBUJACEM', 'RAMCOCEM', 'JKCEMENT', 'SAIL', 'VEDL', 'JINDALSTEL', 'NMDC',
        'NATIONALUM', 'PETRONET', 'GAIL', 'IGL', 'MGL', 'INDIGO', 'DMART', 'TRENT',
        'ABFRL', 'APLAPOLLO', 'ASTRAL', 'SUPREMEIND', 'DIXON', 'VOLTAS', 'BLUESTARCO',
        'WHIRLPOOL', 'BATAINDIA', 'RELAXO', 'CROMPTON', 'SYMPHONY', 'CUMMINSIND',
        'BOSCHLTD', 'MOTHERSON', 'BALKRISIND', 'MRF', 'APOLLOTYRE', 'ESCORTS',
        'ASHOKLEY', 'TIINDIA', 'EXIDEIND', 'BHARATFORG', 'MPHASIS', 'LTTS', 'LTIM',
        'COFORGE', 'PERSISTENT', 'SONATSOFTW', 'ZENSARTECH', 'TATAELXSI', 'KPITTECH',
        'CYIENT', 'LUPIN', 'BIOCON', 'TORNTPHARM', 'AUROPHARMA', 'ALKEM', 'LALPATHLAB',
        'METROPOLIS', 'FORTIS', 'MAXHEALTH', 'HINDPETRO', 'IOC', 'CASTROLIND', 'MRPL',
        'OIL', 'GUJGASLTD', 'GSPL', 'CONCOR', 'IRCTC', 'IRFC', 'RVNL', 'RAILTEL',
        'ADANIPOWER', 'TORNTPOWER', 'NHPC', 'SJVN', 'PTC', 'CESC', 'TATACOMM', 'ZEEL',
        'SAREGAMA', 'JUBLFOOD', 'WESTLIFE', 'DEVYANI', 'GLENMARK', 'IPCALAB',
        'ABBOTINDIA', 'PFIZER', 'GLAXO', 'SANOFI', 'AJANTPHARM', 'GRANULES',
        'NATCOPHARM', 'LAURUSLABS', 'SOLARA', 'ERIS', 'ATUL', 'BALRAMCHIN', 'DEEPAKNTR',
        'GNFC', 'CHAMBLFERT', 'COROMANDEL', 'PIIND', 'ALKYLAMINE', 'SRF', 'NAVINFLUOR',
        'TATACHEM', 'GHCL', 'DCMSHRIRAM', 'JKLAKSHMI', 'STARCEMENT', 'INDIACEM',
        'KAJARIACER', 'CERA', 'POLYCAB', 'KEI', 'VSTIND', 'VGUARD', 'HFCL', 'EMAMILTD',
        'BAJAJCON', 'JYOTHYLAB', 'GILLETTE', 'RADICO', 'GLOBUSSPR', 'GRINDWELL',
        'CARBORUNIV', 'CENTRALBK', 'EQUITASBNK', 'SHRIRAMFIN', 'CHOLAFIN', 'SUNDARMFIN',
        'MUTHOOTFIN', 'MANAPPURAM', 'MOTILALOFS', 'ANGELONE', 'CDSL', 'CAMS', 'MAZDOCK',
        'COCHINSHIP', 'TIMKEN', 'SCHAEFFLER', 'ABB', 'SIEMENS', 'THERMAX', 'BHEL',
        'BEL', 'HAL', 'BEML', 'GRSE', 'MEDANTA', 'KIMS', 'SYNGENE', 'AIAENG',
        'VINATIORGA', 'SUDARSCHEM', 'FINEORG', 'ROSSARI', 'NOCIL', 'CRISIL',
        'CARERATING', 'ICRA', 'TTKPRESTIG', 'WONDERLA', 'LEMONTREE', 'INDHOTEL',
        'MAHLOG', 'TCI', 'ALLCARGO', 'AEGISLOG', 'BLUEDART', 'DELHIVERY', 'VTL',
        'DEEPAKFERT', 'BASF', 'STLTECH', 'TATATECH', 'DATAPATTNS', 'ZOMATO', 'NYKAA',
        'PAYTM', 'POLICYBZR', 'AARTIIND', 'AAVAS', 'ABSLAMC', 'ABCAPITAL', 'ADANIENSOL',
        'ATGL', 'ANURAS', 'APARINDS', 'ASTERDM', 'ASTRAMICRO', 'AUBANK',
        'BAJAJELEC', 'BAJAJHIND', 'BANDHANBNK', 'BAYERCROP', 'BUTTERFLY', 'CANFINHOME',
        'CENTURYPLY', 'CHOLAHLDNG', 'COMPINFO', 'CRAFTSMAN', 'CYIENTDLM', 'DBCORP',
        'DCAL', 'DELHIVERY', 'DEN', 'EASEMYTRIP', 'EIHOTEL', 'ELECON', 'ENDURANCE', 'ENGINERSIN',
        'FEL', 'GABRIEL', 'GALAXYSURF', 'GMMPFAUDLR', 'GODFRYPHLP', 'GODREJIND',
        'GRAPHITE', 'GSFC', 'GTPL', 'GUJALKALI', 'HATSUN', 'HEG', 'HEIDELBERG',
        'HINDCOPPER', 'HINDZINC', 'HMVL', 'HOMEFIRST', 'HSCL', 'IDEA', 'IEX', 'IIFL',
        'IMAGICAA', 'INDIAMART', 'INGERRAND', 'J&KBANK', 'JAGRAN', 'JAICORPLTD',
        'JBCHEPHARM', 'JINDALSAW', 'JINDWORLD', 'JISLJALEQS', 'JKIL', 'JMA', 'JSWHL',
        'JUSTDIAL', 'KALYANKJIL', 'KANSAINER', 'KCP', 'KEC', 'KEYFINSERV', 'KFINTECH',
        'KIRIINDUS', 'KNRCON', 'KRBL', 'LTFOODS', 'LUXIND', 'LXCHEM', 'M&MFIN',
        'MAHAPEXLTD', 'MAHSCOOTER', 'MAHSEAMLES', 'MARKSANS', 'MASFIN', 'MATRIMONY',
        'MAZDA', 'MINDSPACE', 'MOL', 'MONTECARLO', 'MTNL', 'MUNJALAU', 'MUNJALSHOW',
        'NAUKRI', 'NCC', 'NCLIND', 'NDGL', 'NELCAST', 'NETWORK18', 'NFL', 'NIACL',
        'NIITLTD', 'NLCINDIA', 'ONEPOINT', 'ONMOBILE', 'OPTIEMUS', 'ORIENTPPR',
        'PANAMAPET', 'PARAS', 'PGHH', 'POONAWALLA', 'POWERMECH', 'PPAP', 'PPLPHARMA',
        'PRAXIS', 'PVP', 'RAJESHEXPO', 'RAMCOSYS', 'RAMKY', 'RATNAMANI', 'RAYMOND',
        'REDINGTON', 'RENUKA', 'REPCOHOME', 'RESPONIND', 'RHL', 'RICOAUTO', 'RITES',
        'ROUTE', 'RPGLIFE', 'RTNPOWER', 'RUBYMILLS', 'RUCHINFRA', 'RUCHIRA', 'SANGHIIND',
        'SANSERA', 'SARLAPOLY', 'SATIN', 'SCI', 'SETCO', 'SFL', 'SHALBY', 'SHALPAINTS',
        'SHANKARA', 'SHANTIGEAR', 'SHARDACROP', 'SHARDAMOTR', 'SHAREINDIA', 'SHIVALIK',
        'SHREDIGCEM', 'SHREERAMA', 'SHREYANIND', 'SHRIRAMPPS', 'SHYAMCENT', 'SIL',
        'SILGO', 'SIYSIL', 'SKIPPER', 'SNOWMAN', 'SOBHA', 'SOLARINDS', 'SONACOMS',
        'SOUTHBANK', 'SREEL', 'SRHHYPOLTD', 'STARPAPER', 'STARTECK', 'STERTOOLS',
        'STOVEKRAFT', 'SUMMITSEC', 'SUNCLAY', 'SUNDARAM', 'SUNDARMHLD', 'SUNTECK',
        'SUPREMEENG', 'SUPRIYA', 'SURYALAXMI', 'SURYAROSNI', 'SUZLON', 'SWARAJENG',
        'TAJGVK', 'TARAPUR', 'TBZ', 'TCIEXP', 'TDPOWERSYS', 'TECHIN', 'TEJASNET',
        'TERASOFT', 'THANGAMAYL', 'THEINVEST', 'THOMASCOOK', 'THYROCARE', 'TI', 'TIMKEN',
        'TIRUMALCHM', 'TNTELE', 'TOKYOPLAST', 'TOTAL', 'TOUCHWOOD', 'TRIGYN', 'TRIVENI',
        'TVSMOTOR', 'UCOBANK', 'UFLEX', 'UGROCAP', 'UNICHEMLAB', 'UNITY', 'UNIVCABLES',
        'UTIAMC', 'VADILALIND', 'VERTOZ', 'VHL', 'VINDHYATEL', 'VINYLINDIA', 'VISAKAIND',
        'VISASTEEL', 'WEIZMANIND', 'WELENT', 'WIPL', 'WORTH', 'WSTCSTPAPR', 'XELPMOC',
        'XPROINDIA', 'YAARI', 'ZENITHEXPO', 'ZUARI', 'ZYDUSLIFE',
    ]

    nse_tickers = [ticker + '.NS' for ticker in known_tickers]
    print(f"Loaded {len(nse_tickers)} NSE stock symbols")
    return nse_tickers

def validate_ticker(ticker):
    """
    Validate if a ticker has data available in yfinance with rate limiting.
    Returns ticker info if valid, None otherwise.
    """
    try:
        time.sleep(0.05)  # Small delay to avoid rate limiting
        stock = yf.Ticker(ticker)
        info = stock.info

        # Check if we got valid data
        if info and len(info) > 5 and 'symbol' in info:
            # Additional validation - check if it has some price data
            hist = stock.history(period='5d')
            if not hist.empty:
                return {
                    'ticker': ticker,
                    'company_name': info.get('longName', info.get('shortName', '')),
                    'sector': info.get('sector', ''),
                    'industry': info.get('industry', ''),
                    'exchange': 'NSE' if ticker.endswith('.NS') else 'BSE'
                }
    except Exception as e:
        pass

    return None

def get_all_indian_stocks():
    """
    Fetch all Indian stocks from both NSE and BSE.
    Remove duplicates by keeping NSE ticker when a company is listed on both exchanges.
    """
    print("Fetching comprehensive list of Indian stocks from NSE and BSE...")
    print("=" * 70)

    # Get ticker lists
    nse_tickers = get_nse_stock_list()
    bse_tickers = get_bse_stock_codes()

    all_tickers = nse_tickers + bse_tickers
    print(f"\nTotal tickers to validate: {len(all_tickers)}")
    print(f"  NSE tickers: {len(nse_tickers)}")
    print(f"  BSE tickers: {len(bse_tickers)}")
    print("\nValidating tickers (this will take 30-45 minutes)...")
    print("=" * 70)

    # Validate tickers in parallel for faster processing
    valid_stocks = []
    company_to_ticker = {}  # Track companies to remove duplicates

    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        future_to_ticker = {executor.submit(validate_ticker, ticker): ticker
                           for ticker in all_tickers}

        completed = 0
        for future in concurrent.futures.as_completed(future_to_ticker):
            completed += 1
            if completed % 100 == 0:
                print(f"Progress: {completed}/{len(all_tickers)} tickers validated ({completed*100//len(all_tickers)}%)...")

            result = future.result()
            if result:
                company_name = result['company_name']
                ticker = result['ticker']

                # Handle duplicates - prefer NSE over BSE
                if company_name and company_name.strip():
                    if company_name not in company_to_ticker:
                        company_to_ticker[company_name] = result
                        valid_stocks.append(result)
                    elif ticker.endswith('.NS') and company_to_ticker[company_name]['ticker'].endswith('.BO'):
                        # Replace BSE with NSE
                        old_result = company_to_ticker[company_name]
                        valid_stocks.remove(old_result)
                        company_to_ticker[company_name] = result
                        valid_stocks.append(result)
                    # Skip if BSE but NSE already exists
                else:
                    # If no company name, still add
                    valid_stocks.append(result)

    # Create DataFrame
    df = pd.DataFrame(valid_stocks)

    if df.empty:
        print("\nNo valid stocks found!")
        return df

    df = df.rename(columns={
        'ticker': 'Ticker',
        'company_name': 'Company Name',
        'sector': 'Sector',
        'industry': 'Industry',
        'exchange': 'Exchange'
    })

    # Sort by ticker
    df = df.sort_values('Ticker').reset_index(drop=True)

    # Save to CSV
    output_file = 'indian_stocks_tickers.csv'
    df.to_csv(output_file, index=False, encoding='utf-8-sig')

    print("=" * 70)
    print(f"\nSuccessfully validated {len(df)} unique Indian stocks")
    print(f"  NSE stocks: {len(df[df['Exchange'] == 'NSE'])}")
    print(f"  BSE stocks: {len(df[df['Exchange'] == 'BSE'])}")
    print(f"\nSaved to: {output_file}")
    print("\nFirst 10 entries:")
    print(df.head(10).to_string(index=False))
    print("\nLast 10 entries:")
    print(df.tail(10).to_string(index=False))
    print("\nSector distribution:")
    sector_counts = df['Sector'].value_counts().head(10)
    for sector, count in sector_counts.items():
        print(f"  {sector}: {count}")

    print("\nExchange distribution:")
    exchange_counts = df['Exchange'].value_counts()
    for exchange, count in exchange_counts.items():
        print(f"  {exchange}: {count}")

    return df

if __name__ == "__main__":
    df = get_all_indian_stocks()
