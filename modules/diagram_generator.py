"""
UML Diagram Generator Module
Generates key reference diagram in English and 50 variations in Bahasa Indonesia
"""

import json
import random
import copy
from typing import Dict, List, Any


class DiagramGenerator:
    """Generate UML class diagrams for assessment"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # English to Indonesian translation dictionaries
        self.translations = {
            # Class names
            'Customer': 'Pelanggan',
            'Order': 'Pesanan',
            'Product': 'Produk',
            'Payment': 'Pembayaran',
            'ShoppingCart': 'KeranjangBelanja',
            'Invoice': 'Faktur',
            'Shipping': 'Pengiriman',
            
            # Attributes
            'customerId': 'idPelanggan',
            'name': 'nama',
            'email': 'email',
            'phone': 'telepon',
            'address': 'alamat',
            'orderId': 'idPesanan',
            'orderDate': 'tanggalPesanan',
            'totalAmount': 'totalJumlah',
            'status': 'status',
            'productId': 'idProduk',
            'productName': 'namaProduk',
            'price': 'harga',
            'quantity': 'jumlah',
            'description': 'deskripsi',
            'paymentId': 'idPembayaran',
            'paymentMethod': 'metodePembayaran',
            'paymentDate': 'tanggalPembayaran',
            'cartId': 'idKeranjang',
            'invoiceId': 'idFaktur',
            'invoiceNumber': 'nomorFaktur',
            'shippingId': 'idPengiriman',
            'trackingNumber': 'nomorPelacakan',
            
            # Methods
            'register': 'daftar',
            'login': 'masuk',
            'logout': 'keluar',
            'update': 'perbarui',
            'delete': 'hapus',
            'placeOrder': 'buatPesanan',
            'cancelOrder': 'batalkanPesanan',
            'addProduct': 'tambahProduk',
            'removeProduct': 'hapusProduk',
            'processPayment': 'prosesPembayaran',
            'generateInvoice': 'buatFaktur',
            'shipOrder': 'kirimPesanan',
            'trackShipment': 'lacakPengiriman',
            'calculateTotal': 'hitungTotal',
            'applyDiscount': 'terapkanDiskon',
            'validatePayment': 'validasiPembayaran',
            
            # Data types
            'int': 'integer',
            'string': 'teks',
            'float': 'desimal',
            'boolean': 'boolean',
            'date': 'tanggal',
        }
        
        # Synonyms for semantic testing
        self.synonyms = {
            'Pelanggan': ['Konsumen', 'Pembeli', 'Klien'],
            'Pesanan': ['Order', 'Pemesanan'],
            'Produk': ['Barang', 'Item'],
            'Pembayaran': ['Bayar', 'Transaksi'],
            'nama': ['namaLengkap', 'namaPengguna'],
            'harga': ['biaya', 'tarif'],
        }
    
    def generate_key_reference_diagram(self) -> Dict[str, Any]:
        """Generate the base UML diagram in English"""
        
        diagram = {
            "diagram_id": "key_reference_english",
            "language": "english",
            "classes": [
                {
                    "id": "class_1",
                    "name": "Customer",
                    "attributes": [
                        {"name": "customerId", "type": "int", "visibility": "private"},
                        {"name": "name", "type": "string", "visibility": "private"},
                        {"name": "email", "type": "string", "visibility": "private"},
                        {"name": "phone", "type": "string", "visibility": "private"},
                        {"name": "address", "type": "string", "visibility": "private"}
                    ],
                    "methods": [
                        {"name": "register", "parameters": ["email: string", "phone: string"], 
                         "return_type": "boolean", "visibility": "public"},
                        {"name": "login", "parameters": ["email: string"], 
                         "return_type": "boolean", "visibility": "public"},
                        {"name": "update", "parameters": ["field: string", "value: string"], 
                         "return_type": "boolean", "visibility": "public"}
                    ]
                },
                {
                    "id": "class_2",
                    "name": "Order",
                    "attributes": [
                        {"name": "orderId", "type": "int", "visibility": "private"},
                        {"name": "orderDate", "type": "date", "visibility": "private"},
                        {"name": "totalAmount", "type": "float", "visibility": "private"},
                        {"name": "status", "type": "string", "visibility": "private"}
                    ],
                    "methods": [
                        {"name": "placeOrder", "parameters": ["customerId: int"], 
                         "return_type": "boolean", "visibility": "public"},
                        {"name": "cancelOrder", "parameters": [], 
                         "return_type": "boolean", "visibility": "public"},
                        {"name": "calculateTotal", "parameters": [], 
                         "return_type": "float", "visibility": "public"}
                    ]
                },
                {
                    "id": "class_3",
                    "name": "Product",
                    "attributes": [
                        {"name": "productId", "type": "int", "visibility": "private"},
                        {"name": "productName", "type": "string", "visibility": "private"},
                        {"name": "price", "type": "float", "visibility": "private"},
                        {"name": "description", "type": "string", "visibility": "private"}
                    ],
                    "methods": [
                        {"name": "update", "parameters": ["field: string", "value: string"], 
                         "return_type": "boolean", "visibility": "public"},
                        {"name": "applyDiscount", "parameters": ["percentage: float"], 
                         "return_type": "float", "visibility": "public"}
                    ]
                },
                {
                    "id": "class_4",
                    "name": "Payment",
                    "attributes": [
                        {"name": "paymentId", "type": "int", "visibility": "private"},
                        {"name": "paymentMethod", "type": "string", "visibility": "private"},
                        {"name": "paymentDate", "type": "date", "visibility": "private"},
                        {"name": "totalAmount", "type": "float", "visibility": "private"}
                    ],
                    "methods": [
                        {"name": "processPayment", "parameters": ["orderId: int"], 
                         "return_type": "boolean", "visibility": "public"},
                        {"name": "validatePayment", "parameters": [], 
                         "return_type": "boolean", "visibility": "public"}
                    ]
                },
                {
                    "id": "class_5",
                    "name": "ShoppingCart",
                    "attributes": [
                        {"name": "cartId", "type": "int", "visibility": "private"},
                        {"name": "customerId", "type": "int", "visibility": "private"},
                        {"name": "totalAmount", "type": "float", "visibility": "private"}
                    ],
                    "methods": [
                        {"name": "addProduct", "parameters": ["productId: int", "quantity: int"], 
                         "return_type": "boolean", "visibility": "public"},
                        {"name": "removeProduct", "parameters": ["productId: int"], 
                         "return_type": "boolean", "visibility": "public"},
                        {"name": "calculateTotal", "parameters": [], 
                         "return_type": "float", "visibility": "public"}
                    ]
                },
                {
                    "id": "class_6",
                    "name": "Invoice",
                    "attributes": [
                        {"name": "invoiceId", "type": "int", "visibility": "private"},
                        {"name": "invoiceNumber", "type": "string", "visibility": "private"},
                        {"name": "orderId", "type": "int", "visibility": "private"},
                        {"name": "totalAmount", "type": "float", "visibility": "private"}
                    ],
                    "methods": [
                        {"name": "generateInvoice", "parameters": ["orderId: int"], 
                         "return_type": "boolean", "visibility": "public"}
                    ]
                },
                {
                    "id": "class_7",
                    "name": "Shipping",
                    "attributes": [
                        {"name": "shippingId", "type": "int", "visibility": "private"},
                        {"name": "orderId", "type": "int", "visibility": "private"},
                        {"name": "trackingNumber", "type": "string", "visibility": "private"},
                        {"name": "status", "type": "string", "visibility": "private"}
                    ],
                    "methods": [
                        {"name": "shipOrder", "parameters": ["orderId: int"], 
                         "return_type": "boolean", "visibility": "public"},
                        {"name": "trackShipment", "parameters": [], 
                         "return_type": "string", "visibility": "public"}
                    ]
                }
            ],
            "relationships": [
                {
                    "id": "rel_1",
                    "type": "association",
                    "source": "class_1",
                    "target": "class_2",
                    "multiplicity_source": "1",
                    "multiplicity_target": "*",
                    "label": "places"
                },
                {
                    "id": "rel_2",
                    "type": "association",
                    "source": "class_2",
                    "target": "class_3",
                    "multiplicity_source": "1",
                    "multiplicity_target": "*",
                    "label": "contains"
                },
                {
                    "id": "rel_3",
                    "type": "association",
                    "source": "class_2",
                    "target": "class_4",
                    "multiplicity_source": "1",
                    "multiplicity_target": "1",
                    "label": "paid by"
                },
                {
                    "id": "rel_4",
                    "type": "composition",
                    "source": "class_1",
                    "target": "class_5",
                    "multiplicity_source": "1",
                    "multiplicity_target": "1",
                    "label": "has"
                },
                {
                    "id": "rel_5",
                    "type": "aggregation",
                    "source": "class_5",
                    "target": "class_3",
                    "multiplicity_source": "1",
                    "multiplicity_target": "*",
                    "label": "contains"
                },
                {
                    "id": "rel_6",
                    "type": "association",
                    "source": "class_2",
                    "target": "class_6",
                    "multiplicity_source": "1",
                    "multiplicity_target": "1",
                    "label": "generates"
                },
                {
                    "id": "rel_7",
                    "type": "association",
                    "source": "class_2",
                    "target": "class_7",
                    "multiplicity_source": "1",
                    "multiplicity_target": "1",
                    "label": "shipped via"
                },
                {
                    "id": "rel_8",
                    "type": "association",
                    "source": "class_4",
                    "target": "class_6",
                    "multiplicity_source": "1",
                    "multiplicity_target": "1",
                    "label": "documented in"
                }
            ]
        }
        
        return diagram
    
    def translate_to_indonesian(self, diagram: Dict[str, Any]) -> Dict[str, Any]:
        """Translate entire diagram to Indonesian"""
        translated = copy.deepcopy(diagram)
        translated['language'] = 'bahasa_indonesia'
        
        for cls in translated['classes']:
            # Translate class name
            cls['name'] = self.translations.get(cls['name'], cls['name'])
            
            # Translate attributes
            for attr in cls['attributes']:
                attr['name'] = self.translations.get(attr['name'], attr['name'])
                attr['type'] = self.translations.get(attr['type'], attr['type'])
            
            # Translate methods
            for method in cls['methods']:
                method['name'] = self.translations.get(method['name'], method['name'])
                # Translate parameter types
                translated_params = []
                for param in method['parameters']:
                    parts = param.split(':')
                    if len(parts) == 2:
                        param_name = parts[0].strip()
                        param_type = parts[1].strip()
                        param_name_trans = self.translations.get(param_name, param_name)
                        param_type_trans = self.translations.get(param_type, param_type)
                        translated_params.append(f"{param_name_trans}: {param_type_trans}")
                    else:
                        translated_params.append(param)
                method['parameters'] = translated_params
                method['return_type'] = self.translations.get(method['return_type'], method['return_type'])
        
        return translated
    
    def introduce_typos(self, text: str) -> str:
        """Introduce spelling errors"""
        if len(text) < 4:
            return text
        
        # Common typo patterns
        typo_patterns = [
            ('a', 'e'),
            ('e', 'a'),
            ('i', 'e'),
            ('ng', 'n'),
            ('an', 'en'),
        ]
        
        pattern = random.choice(typo_patterns)
        if pattern[0] in text:
            return text.replace(pattern[0], pattern[1], 1)
        return text
    
    def generate_student_variations(self, key_diagram: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate 50 student diagram variations"""
        variations = []
        variation_config = self.config['variation_distribution']
        diagram_counter = 1
        
        # 1. Perfect Translation (5 diagrams)
        for i in range(variation_config['perfect_translation']):
            var = self.translate_to_indonesian(key_diagram)
            var['diagram_id'] = f"student_{diagram_counter:03d}"
            var['variation_type'] = 'perfect_translation'
            var['expected_score_range'] = [95, 100]
            var['modifications'] = ['Translated all elements to Indonesian']
            variations.append(var)
            diagram_counter += 1
        
        # 2. Minor Semantic Variations (10 diagrams)
        for i in range(variation_config['minor_semantic']):
            var = self.translate_to_indonesian(key_diagram)
            var['diagram_id'] = f"student_{diagram_counter:03d}"
            var['variation_type'] = 'minor_semantic'
            var['expected_score_range'] = [85, 95]
            modifications = []
            
            # Apply synonyms to some class names
            num_synonyms = random.randint(1, 3)
            for cls in random.sample(var['classes'], min(num_synonyms, len(var['classes']))):
                if cls['name'] in self.synonyms:
                    original = cls['name']
                    cls['name'] = random.choice(self.synonyms[original])
                    modifications.append(f"Changed {original} to {cls['name']} (synonym)")
            
            var['modifications'] = modifications
            variations.append(var)
            diagram_counter += 1
        
        # 3. Spelling Errors (5 diagrams)
        for i in range(variation_config['spelling_errors']):
            var = self.translate_to_indonesian(key_diagram)
            var['diagram_id'] = f"student_{diagram_counter:03d}"
            var['variation_type'] = 'spelling_errors'
            var['expected_score_range'] = [80, 90]
            modifications = []
            
            # Introduce typos in 2-3 class names
            num_typos = random.randint(2, 3)
            for cls in random.sample(var['classes'], min(num_typos, len(var['classes']))):
                original = cls['name']
                cls['name'] = self.introduce_typos(cls['name'])
                modifications.append(f"Typo: {original} → {cls['name']}")
            
            var['modifications'] = modifications
            variations.append(var)
            diagram_counter += 1
        
        # 4. Missing Classes (5 diagrams)
        for i in range(variation_config['missing_classes']):
            var = self.translate_to_indonesian(key_diagram)
            var['diagram_id'] = f"student_{diagram_counter:03d}"
            var['variation_type'] = 'missing_classes'
            var['expected_score_range'] = [60, 80]
            modifications = []
            
            # Remove 1-2 classes
            num_remove = random.randint(1, 2)
            classes_to_remove = random.sample(var['classes'], num_remove)
            removed_ids = [cls['id'] for cls in classes_to_remove]
            
            for cls in classes_to_remove:
                var['classes'].remove(cls)
                modifications.append(f"Removed class: {cls['name']}")
            
            # Remove relationships involving removed classes
            var['relationships'] = [
                rel for rel in var['relationships']
                if rel['source'] not in removed_ids and rel['target'] not in removed_ids
            ]
            
            var['modifications'] = modifications
            variations.append(var)
            diagram_counter += 1
        
        # 5. Extra Classes (5 diagrams)
        for i in range(variation_config['extra_classes']):
            var = self.translate_to_indonesian(key_diagram)
            var['diagram_id'] = f"student_{diagram_counter:03d}"
            var['variation_type'] = 'extra_classes'
            var['expected_score_range'] = [60, 80]
            modifications = []
            
            # Add 1-2 extra classes
            num_add = random.randint(1, 2)
            for j in range(num_add):
                new_class_id = f"class_extra_{j+1}"
                new_class = {
                    "id": new_class_id,
                    "name": f"ExtraClass{j+1}",
                    "attributes": [
                        {"name": "extraAttr", "type": "teks", "visibility": "private"}
                    ],
                    "methods": [
                        {"name": "extraMethod", "parameters": [], "return_type": "boolean", "visibility": "public"}
                    ]
                }
                var['classes'].append(new_class)
                modifications.append(f"Added extra class: {new_class['name']}")
            
            var['modifications'] = modifications
            variations.append(var)
            diagram_counter += 1
        
        # 6. Missing Attributes (5 diagrams)
        for i in range(variation_config['missing_attributes']):
            var = self.translate_to_indonesian(key_diagram)
            var['diagram_id'] = f"student_{diagram_counter:03d}"
            var['variation_type'] = 'missing_attributes'
            var['expected_score_range'] = [70, 85]
            modifications = []
            
            # Remove 2-4 attributes across classes
            num_remove = random.randint(2, 4)
            classes_with_attrs = [cls for cls in var['classes'] if len(cls['attributes']) > 1]
            
            for _ in range(min(num_remove, len(classes_with_attrs))):
                cls = random.choice(classes_with_attrs)
                if len(cls['attributes']) > 1:
                    removed_attr = cls['attributes'].pop(random.randint(0, len(cls['attributes'])-1))
                    modifications.append(f"Removed attribute {removed_attr['name']} from {cls['name']}")
            
            var['modifications'] = modifications
            variations.append(var)
            diagram_counter += 1
        
        # 7. Missing Methods (5 diagrams)
        for i in range(variation_config['missing_methods']):
            var = self.translate_to_indonesian(key_diagram)
            var['diagram_id'] = f"student_{diagram_counter:03d}"
            var['variation_type'] = 'missing_methods'
            var['expected_score_range'] = [70, 85]
            modifications = []
            
            # Remove 2-4 methods across classes
            num_remove = random.randint(2, 4)
            classes_with_methods = [cls for cls in var['classes'] if len(cls['methods']) > 1]
            
            for _ in range(min(num_remove, len(classes_with_methods))):
                cls = random.choice(classes_with_methods)
                if len(cls['methods']) > 1:
                    removed_method = cls['methods'].pop(random.randint(0, len(cls['methods'])-1))
                    modifications.append(f"Removed method {removed_method['name']} from {cls['name']}")
            
            var['modifications'] = modifications
            variations.append(var)
            diagram_counter += 1
        
        # 8. Wrong Relationships (5 diagrams)
        for i in range(variation_config['wrong_relationships']):
            var = self.translate_to_indonesian(key_diagram)
            var['diagram_id'] = f"student_{diagram_counter:03d}"
            var['variation_type'] = 'wrong_relationships'
            var['expected_score_range'] = [50, 70]
            modifications = []
            
            # Change 2-3 relationship types
            num_change = random.randint(2, 3)
            rel_types = ['association', 'generalization', 'aggregation', 'composition']
            
            for rel in random.sample(var['relationships'], min(num_change, len(var['relationships']))):
                original_type = rel['type']
                new_type = random.choice([t for t in rel_types if t != original_type])
                rel['type'] = new_type
                modifications.append(f"Changed relationship {rel['id']} from {original_type} to {new_type}")
            
            var['modifications'] = modifications
            variations.append(var)
            diagram_counter += 1
        
        # 9. Combination Errors (5 diagrams)
        for i in range(variation_config['combination_errors']):
            var = self.translate_to_indonesian(key_diagram)
            var['diagram_id'] = f"student_{diagram_counter:03d}"
            var['variation_type'] = 'combination_errors'
            var['expected_score_range'] = [40, 60]
            modifications = []
            
            # Remove 1 class
            if var['classes']:
                removed_class = var['classes'].pop(random.randint(0, len(var['classes'])-1))
                removed_id = removed_class['id']
                modifications.append(f"Removed class: {removed_class['name']}")
                
                # Remove related relationships
                var['relationships'] = [
                    rel for rel in var['relationships']
                    if rel['source'] != removed_id and rel['target'] != removed_id
                ]
            
            # Remove some attributes
            for cls in random.sample(var['classes'], min(2, len(var['classes']))):
                if cls['attributes']:
                    removed_attr = cls['attributes'].pop(0)
                    modifications.append(f"Removed attribute {removed_attr['name']} from {cls['name']}")
            
            # Change relationship types
            for rel in random.sample(var['relationships'], min(2, len(var['relationships']))):
                rel_types = ['association', 'generalization', 'aggregation', 'composition']
                original_type = rel['type']
                new_type = random.choice([t for t in rel_types if t != original_type])
                rel['type'] = new_type
                modifications.append(f"Changed relationship {rel['id']} to {new_type}")
            
            var['modifications'] = modifications
            variations.append(var)
            diagram_counter += 1
        
        return variations
    
    def save_diagrams(self, key_diagram: Dict[str, Any], student_diagrams: List[Dict[str, Any]], 
                     data_dir: str = 'data'):
        """Save diagrams to JSON files"""
        import os
        
        # Create directories
        os.makedirs(data_dir, exist_ok=True)
        os.makedirs(f"{data_dir}/student_diagrams", exist_ok=True)
        
        # Save key reference
        with open(f"{data_dir}/key_reference.json", 'w', encoding='utf-8') as f:
            json.dump(key_diagram, f, indent=2, ensure_ascii=False)
        
        # Save each student diagram
        for diagram in student_diagrams:
            filename = f"{data_dir}/student_diagrams/{diagram['diagram_id']}.json"
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(diagram, f, indent=2, ensure_ascii=False)
        
        print(f"Saved key reference and {len(student_diagrams)} student diagrams to {data_dir}/")
