from core.connectors.SAXO import SaxoOrderInterface

required = {'CHFJPY':
                {
                    'updatable_time': 60,
                    'supplier': SaxoOrderInterface
                },
            'CHFUSD':
                {
                    'updatable_time': 60,
                    'supplier': SaxoOrderInterface
                }
            }