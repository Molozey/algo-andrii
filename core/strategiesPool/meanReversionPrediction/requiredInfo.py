from core.dataSuppliers.SAXO import SaxoDataProvider

required = {'CHFJPY':
                {
                    'updatable_time': 60,
                    'supplier': SaxoDataProvider
                },
            'EURGBP':
                {
                    'updatable_time': 60,
                    'supplier': SaxoDataProvider
                }
            }