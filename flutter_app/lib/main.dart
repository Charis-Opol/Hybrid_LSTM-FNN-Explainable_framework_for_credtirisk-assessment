import 'dart:convert';
import 'dart:io';

import 'package:file_picker/file_picker.dart';
import 'package:flutter/material.dart';
import 'package:http/http.dart' as http;
import 'package:intl/intl.dart';

void main() => runApp(const RiskApp());

class RiskApp extends StatelessWidget {
  const RiskApp({super.key});
  @override
  Widget build(BuildContext context) => MaterialApp(
        title: 'RiskLens Uganda',
        debugShowCheckedModeBanner: false,
        theme: ThemeData(
          useMaterial3: true,
          colorScheme: ColorScheme.fromSeed(seedColor: const Color(0xFF0B6E4F)),
          scaffoldBackgroundColor: const Color(0xFFF7F9F8),
          inputDecorationTheme: const InputDecorationTheme(border: OutlineInputBorder()),
        ),
        home: const AssessmentPage(),
      );
}

class AssessmentPage extends StatefulWidget {
  const AssessmentPage({super.key});
  @override
  State<AssessmentPage> createState() => _AssessmentPageState();
}

class _AssessmentPageState extends State<AssessmentPage> {
  final _formKey = GlobalKey<FormState>();
  final _api = TextEditingController(text: 'http://10.0.2.2:8000');
  final _borrower = TextEditingController();
  final _loanAmount = TextEditingController();
  PlatformFile? _file;
  Map<String, dynamic>? _result;
  String? _error;
  bool _loading = false;

  @override
  void dispose() {
    _api.dispose(); _borrower.dispose(); _loanAmount.dispose();
    super.dispose();
  }

  Future<void> _chooseFile() async {
    final selection = await FilePicker.platform.pickFiles(type: FileType.custom, allowedExtensions: ['csv'], withData: false);
    if (selection != null) setState(() { _file = selection.files.single; _error = null; });
  }

  Future<void> _assess() async {
    if (!_formKey.currentState!.validate()) return;
    if (_file?.path == null) { setState(() => _error = 'Choose the transaction CSV first.'); return; }
    setState(() { _loading = true; _error = null; _result = null; });
    try {
      final request = http.MultipartRequest('POST', Uri.parse('${_api.text.trim()}/assess'))
        ..fields.addAll({
          'borrower_id': _borrower.text.trim(),
          'loan_amount': _loanAmount.text.trim(),
          'income_source': 'Unknown', 'sacco_membership': 'Unknown', 'location': 'Unknown',
          'preferred_network': 'Unknown', 'preferred_channel': 'Unknown',
        })
        ..files.add(await http.MultipartFile.fromPath('transactions', _file!.path!));
      final response = await request.send();
      final body = await response.stream.bytesToString();
      final decoded = jsonDecode(body) as Map<String, dynamic>;
      if (response.statusCode >= 300) throw Exception(decoded['detail'] ?? 'Unable to assess this borrower.');
      if (mounted) setState(() => _result = decoded);
    } on SocketException {
      setState(() => _error = 'Cannot reach the API. Start it first and check the API address.');
    } catch (e) {
      setState(() => _error = e.toString().replaceFirst('Exception: ', ''));
    } finally {
      if (mounted) setState(() => _loading = false);
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('RiskLens Uganda'), backgroundColor: const Color(0xFF0B6E4F), foregroundColor: Colors.white),
      body: SafeArea(child: Center(child: ConstrainedBox(
        constraints: const BoxConstraints(maxWidth: 780),
        child: ListView(padding: const EdgeInsets.all(20), children: [
          const Text('Mobile money credit-risk assessment', style: TextStyle(fontSize: 27, fontWeight: FontWeight.bold)),
          const SizedBox(height: 8),
          const Text('Upload one borrower’s 12-month transaction history. The hybrid LSTM–FNN model returns a decision-support risk estimate.'),
          const SizedBox(height: 24),
          Form(key: _formKey, child: Column(children: [
            TextFormField(controller: _api, decoration: const InputDecoration(labelText: 'API address'), validator: (v) => Uri.tryParse(v ?? '')?.hasAbsolutePath == true ? null : 'Enter a valid API URL'),
            const SizedBox(height: 14),
            Row(children: [Expanded(child: TextFormField(controller: _borrower, decoration: const InputDecoration(labelText: 'Borrower ID'), validator: (v) => (v ?? '').trim().isEmpty ? 'Required' : null)), const SizedBox(width: 14), Expanded(child: TextFormField(controller: _loanAmount, keyboardType: TextInputType.number, decoration: const InputDecoration(labelText: 'Requested loan amount (UGX)'), validator: (v) => double.tryParse(v ?? '') == null ? 'Enter a number' : null))]),
          ])),
          const SizedBox(height: 18),
          OutlinedButton.icon(onPressed: _chooseFile, icon: const Icon(Icons.upload_file), label: Text(_file == null ? 'Choose transactions CSV' : _file!.name)),
          const Padding(padding: EdgeInsets.only(top: 8), child: Text('Required columns: borrower_id, transaction_date, transaction_amount. Optional: transaction_type, balance.')), 
          const SizedBox(height: 18),
          FilledButton.icon(onPressed: _loading ? null : _assess, icon: _loading ? const SizedBox.square(dimension: 18, child: CircularProgressIndicator(strokeWidth: 2)) : const Icon(Icons.analytics), label: const Text('Assess default risk')),
          if (_error != null) Padding(padding: const EdgeInsets.only(top: 16), child: Text(_error!, style: TextStyle(color: Theme.of(context).colorScheme.error))),
          if (_result != null) _ResultCard(result: _result!),
          const SizedBox(height: 28),
          const Card(child: Padding(padding: EdgeInsets.all(16), child: Text('Important: This app is for decision support only. Review the borrower context and adverse-action, fairness, privacy, and regulatory requirements before any lending decision.'))),
        ]),
      ))),
    );
  }
}

class _ResultCard extends StatelessWidget {
  const _ResultCard({required this.result});
  final Map<String, dynamic> result;
  @override
  Widget build(BuildContext context) {
    final risk = (result['default_probability'] as num).toDouble();
    final defaultLikely = result['decision'] == 'Likely to default';
    final color = defaultLikely ? Colors.deepOrange : const Color(0xFF0B6E4F);
    return Card(margin: const EdgeInsets.only(top: 20), child: Padding(padding: const EdgeInsets.all(20), child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
      Text('Assessment result', style: Theme.of(context).textTheme.titleLarge), const SizedBox(height: 14),
      Text('${NumberFormat.percentPattern().format(risk)} default probability', style: TextStyle(fontSize: 30, fontWeight: FontWeight.bold, color: color)),
      const SizedBox(height: 8), Text(result['decision'] as String, style: TextStyle(fontWeight: FontWeight.w600, color: color)),
      const Divider(height: 28),
      Text('${result['transactions_used']} transactions • ${result['period']['from']} to ${result['period']['to']}'),
      Text('Decision threshold: ${NumberFormat.percentPattern().format((result['threshold'] as num).toDouble())}'),
      const SizedBox(height: 10), Text(result['notice'] as String, style: Theme.of(context).textTheme.bodySmall),
    ]));
  }
}
